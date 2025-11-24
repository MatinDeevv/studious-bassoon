import sys
import time
import random
import threading
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ccxt

try:
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                   QLabel, QPushButton, QComboBox, QTextEdit, QFrame, QTabWidget,
                                   QProgressBar, QMessageBox, QLineEdit, QFormLayout, QSplitter, QCheckBox)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("PySide6 not installed! Run: pip install PySide6 PySide6-Addons")
    sys.exit(1)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os

class BotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor: 3 actions → Long / Short / Flat
        self.actor_action = nn.Linear(hidden_dim, 3)
        # Actor heads for leverage and risk (continuous 0–1 → will be scaled later)
        self.actor_lever = nn.Linear(hidden_dim, 1)
        self.actor_risk = nn.Linear(hidden_dim, 1)
        # Critic: value estimate
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 1:           # in case a single vector is passed (e.g. live inference)
            x = x.unsqueeze(0)
        features = self.shared(x)
        action_logits = self.actor_action(features)           # shape: (batch, 3)
        lever_raw     = self.actor_lever(features)            # shape: (batch, 1)
        risk_raw      = self.actor_risk(features)             # shape: (batch, 1)
        value         = self.critic(features)                 # shape: (batch, 1)
        return action_logits, lever_raw.squeeze(-1), risk_raw.squeeze(-1), value.squeeze(-1)
class AppConfig:
    def __init__(self):
        self.symbol = "ETHUSDT"
        self.timeframe = "3m"
        self.csv_path = "eth_3m_ohlcv.csv"
        self.population_size = 24
        self.senate_size = 5
        self.mutation_scale = 0.05
        self.max_leverage = 10.0
        self.train_len = 2000
        self.test_len = 1000
        self.episode_len = 256
        self.train_sleep = 0.0
        self.live_mode = True
        self.replay_speed = 0.0
        self.fetch_limit = 5000
        self.max_generations = 100
        self.overfit_threshold = 1.5
        self.real_trade_mode = False
        self.rl_epochs = 10
        self.rl_batch_size = 64
        self.rl_lr = 0.0003
        self.rl_gamma = 0.99
        self.rl_gae_lambda = 0.95
        self.rl_clip_ratio = 0.2
        self.rl_value_coef = 0.5
        self.rl_entropy_coef = 0.01


class SharedState:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.lock = threading.Lock()
        self.running = True
        self.feature_dim = None
        self.latest_obs = None
        self.engine_status = None
        self.best_bots = []
        self.live_signal = None
        self.last_action = None
        self.equity_curves = {}
        self.price_history = []
        self.gen_stats = []

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False

    def set_feature_dim(self, d):
        with self.lock:
            self.feature_dim = d

    def update_latest_obs(self, obs):
        with self.lock:
            self.latest_obs = obs

    def get_latest_obs(self):
        with self.lock:
            return self.latest_obs

    def update_engine_status(self, g, b, w):
        with self.lock:
            self.engine_status = (g, b, w)

    def get_engine_status(self):
        with self.lock:
            return self.engine_status

    def update_best_bots(self, bots):
        with self.lock:
            self.best_bots = bots

    def get_best_bots(self):
        with self.lock:
            return self.best_bots

    def update_live_signal(self, signal):
        with self.lock:
            self.live_signal = signal

    def get_live_signal(self):
        with self.lock:
            return self.live_signal

    def update_last_action(self, payload):
        with self.lock:
            self.last_action = payload

    def get_last_action(self):
        with self.lock:
            return self.last_action

    def update_equity_curve(self, bot_id, curve):
        with self.lock:
            self.equity_curves[bot_id] = curve

    def get_equity_curve(self, bot_id):
        with self.lock:
            return self.equity_curves.get(bot_id, [])

    def update_price_history(self, price):
        with self.lock:
            self.price_history.append(price)
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]

    def get_price_history(self):
        with self.lock:
            return self.price_history[:]

    def update_gen_stats(self, stats):
        with self.lock:
            self.gen_stats.append(stats)

    def get_gen_stats(self):
        with self.lock:
            return self.gen_stats[:]


def create_logger(cfg, log_widget=None):
    logger = logging.getLogger("quant_senate")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(ch)
    if log_widget:
        class QtHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                log_widget.append(msg)

        qt_handler = QtHandler()
        qt_handler.setFormatter(fmt)
        logger.addHandler(qt_handler)
    return logger


class DataManager:
    def __init__(self, cfg, state, logger):
        self.cfg = cfg
        self.state = state
        self.logger = logger
        self.symbol = self.cfg.symbol.replace('/', '')
        self.timeframe = self.cfg.timeframe
        self.train_len = self.cfg.train_len
        self.test_len = self.cfg.test_len
        self.episode_len = self.cfg.episode_len
        self.min_history = self.train_len + self.test_len + 2 * self.episode_len
        self.df = None
        self.feature_matrix = None
        self.price_vector = None
        self.feature_dim = 0
        self.train_idx = None
        self.test_idx = None
        self.exchange = ccxt.binance()
        self.last_timestamp = 0
        self._load_or_fetch()
        self.ema_fast_state = 0
        self.ema_slow_state = 0

    def _load_or_fetch(self):
        path = self.cfg.csv_path
        try:
            self.df = pd.read_csv(path)
            self.logger.info("Loaded CSV")
            if len(self.df) < self.min_history:
                self.logger.warning("CSV has insufficient data; fetching more")
                self._fetch_additional_data(older=True)
        except Exception as e:
            self.logger.warning(f"CSV not found or corrupted ({e}); fetching historical data")
            self._fetch_more_data()

        if self.df is None or self.df.empty:
            return

        # -------------------------------------------------
        # FIX: handle both string and datetime timestamp columns
        # -------------------------------------------------
        if 'timestamp' in self.df.columns:
            if self.df['timestamp'].dtype == 'object':  # it's stored as string
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
            elif not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
                # fallback: try to convert whatever it is
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True, errors='coerce')

            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            # now safe to get the last timestamp as milliseconds
            self.last_timestamp = int(self.df['timestamp'].iloc[-1].timestamp() * 1000)
        else:
            # very old CSV without timestamp column – just use current time
            self.last_timestamp = self.exchange.milliseconds()

        # -------------------------------------------------
        # verify required OHLCV columns exist
        # -------------------------------------------------
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in self.df.columns:
                self.logger.error(f"Missing column: {c}")
                self.df = None
                return

        self._build_features()
        self.state.update_price_history(self.df['close'].iloc[-1])

    def _fetch_more_data(self):
        ohlcv = []
        limit = 1000
        interval_ms = 3 * 60 * 1000
        since = self.exchange.milliseconds() - (self.min_history * interval_ms)
        while len(ohlcv) < self.min_history:
            new_ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=limit)
            if not new_ohlcv:
                break
            ohlcv.extend(new_ohlcv)
            since = new_ohlcv[-1][0] + 1
            ohlcv = sorted(ohlcv, key=lambda x: x[0])
            ohlcv = list(dict.fromkeys(tuple(row) for row in ohlcv))
        if len(ohlcv) < self.min_history:
            self.logger.error("Unable to fetch enough data")
            self.df = None
            return
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms', utc=True)
        self.df.to_csv(self.cfg.csv_path, index=False)
        self.logger.info(f"Fetched and saved {len(ohlcv)} bars")

    def _fetch_additional_data(self, older=True):
        if older:
            earliest_ts = self.df['timestamp'].iloc[0].timestamp() * 1000
            needed = self.min_history - len(self.df)
            interval_ms = 3 * 60 * 1000
            since = earliest_ts - (needed * interval_ms) - interval_ms
            limit = 1000
            additional_ohlcv = []
            while len(additional_ohlcv) < needed:
                new = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=limit)
                if not new:
                    break
                additional_ohlcv.extend(new)
                since = new[-1][0] + 1
            if additional_ohlcv:
                additional_df = pd.DataFrame(additional_ohlcv,
                                             columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                additional_df['timestamp'] = pd.to_datetime(additional_df['timestamp'], unit='ms', utc=True)
                self.df = pd.concat([additional_df, self.df], ignore_index=True)
                self.df = self.df.drop_duplicates(subset=['timestamp'])
                self.df = self.df.sort_values('timestamp')
                self.df.to_csv(self.cfg.csv_path, index=False)
                self.logger.info(f"Added {len(additional_ohlcv)} older bars")
        if len(self.df) < self.min_history:
            self.logger.error("Still insufficient data after fetch")
            self.df = None

    def _ema(self, x, period):
        alpha = 2.0 / (period + 1.0)
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    def _ema_incremental(self, x, period, last_ema):
        alpha = 2.0 / (period + 1.0)
        y = np.zeros_like(x)
        if len(x) == 0:
            return y
        y[0] = alpha * x[0] + (1 - alpha) * last_ema
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    def _rsi(self, x, period):
        diff = np.diff(x, prepend=x[0])
        gain = np.where(diff > 0, diff, 0.0)
        loss = np.where(diff < 0, -diff, 0.0)
        roll_up = self._ema(gain, period)
        roll_down = self._ema(loss, period)
        rs = roll_up / (roll_down + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _macd(self, x, fast, slow, signal):
        ema_fast = self._ema(x, fast)
        ema_slow = self._ema(x, slow)
        macd = ema_fast - ema_slow
        signal_line = self._ema(macd, signal)
        hist = macd - signal_line
        return macd, signal_line, hist

    def _bb(self, x, period, std_dev):
        ma = self._ema(x, period)
        variance = self._ema((x - ma) ** 2, period)
        std = np.sqrt(variance)
        upper = ma + std * std_dev
        lower = ma - std * std_dev
        return upper, lower

    def _atr(self, high, low, close, period):
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        atr = self._ema(tr, period)
        return atr

    def _build_features(self, incremental=False, new_data=None):
        if incremental and new_data is not None:
            close = new_data['close'].values.astype(float)
            high = new_data['high'].values.astype(float)
            low = new_data['low'].values.astype(float)
            volume = new_data['volume'].values.astype(float)
            ret = np.zeros_like(close)
            if len(close) > 1:
                ret[1:] = (close[1:] - close[:-1]) / close[:-1]
            ma_fast = self._ema_incremental(close, 20, self.ema_fast_state)
            self.ema_fast_state = ma_fast[-1] if len(ma_fast) > 0 else self.ema_fast_state
            ma_slow = self._ema_incremental(close, 50, self.ema_slow_state)
            self.ema_slow_state = ma_slow[-1] if len(ma_slow) > 0 else self.ema_slow_state
            rsi = self._rsi(close, 14)
            vol_norm = (volume - volume.mean()) / (volume.std() + 1e-8)
            macd, signal, hist = self._macd(close, 12, 26, 9)
            bb_upper, bb_lower = self._bb(close, 20, 2)
            atr = self._atr(high, low, close, 14)
            new_feats = np.stack([
                close / np.maximum(close.mean(), 1e-6),
                ret,
                (ma_fast - close) / np.maximum(close, 1e-6),
                (ma_slow - close) / np.maximum(close, 1e-6),
                rsi / 100.0,
                vol_norm,
                macd / close.mean(),
                hist / close.mean(),
                (bb_upper - close) / close,
                (close - bb_lower) / close,
                atr / close.mean()
            ], axis=1)
            self.feature_matrix = np.vstack(
                (self.feature_matrix, new_feats)) if self.feature_matrix is not None else new_feats
            self.price_vector = np.hstack((self.price_vector, close)) if self.price_vector is not None else close
            self.state.update_price_history(close[-1])
        else:
            df = self.df.copy()
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            volume = df["volume"].values.astype(float)
            ret = np.zeros_like(close)
            ret[1:] = (close[1:] - close[:-1]) / close[:-1]
            ma_fast = self._ema(close, 20)
            self.ema_fast_state = ma_fast[-1] if len(ma_fast) > 0 else 0
            ma_slow = self._ema(close, 50)
            self.ema_slow_state = ma_slow[-1] if len(ma_slow) > 0 else 0
            rsi = self._rsi(close, 14)
            vol_norm = (volume - volume.mean()) / (volume.std() + 1e-8)
            macd, signal, hist = self._macd(close, 12, 26, 9)
            bb_upper, bb_lower = self._bb(close, 20, 2)
            atr = self._atr(high, low, close, 14)
            feats = np.stack([
                close / np.maximum(close.mean(), 1e-6),
                ret,
                (ma_fast - close) / np.maximum(close, 1e-6),
                (ma_slow - close) / np.maximum(close, 1e-6),
                rsi / 100.0,
                vol_norm,
                macd / close.mean(),
                hist / close.mean(),
                (bb_upper - close) / close,
                (close - bb_lower) / close,
                atr / close.mean()
            ], axis=1)
            self.feature_matrix = feats
            self.price_vector = close
            self.state.update_price_history(close[-1])
        self.feature_dim = self.feature_matrix.shape[1]
        n = self.feature_matrix.shape[0]
        if n < self.min_history:
            self.logger.error(f"Not enough history: have {n}, need {self.min_history}")
            self.train_idx = None
            self.test_idx = None
            return
        start = max(0, n - (self.train_len + self.test_len))
        self.train_idx = (start, start + self.train_len)
        self.test_idx = (start + self.train_len, n)
        self.state.set_feature_dim(self.feature_dim)

    def sample_episode(self, split):
        if self.train_idx is None or self.test_idx is None:
            return None, None
        if split == "train":
            lo, hi = self.train_idx
        else:
            lo, hi = self.test_idx
        length = hi - lo
        if length < self.episode_len + 1:
            return None, None
        start = np.random.randint(lo, hi - self.episode_len)
        end = start + self.episode_len
        obs = self.feature_matrix[start:end]
        price = self.price_vector[start:end]
        return obs, price

    def append_new_data(self, new_ohlcv):
        if not new_ohlcv:
            return
        new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df.to_csv(self.cfg.csv_path, index=False)
        self.last_timestamp = new_df['timestamp'].iloc[-1].timestamp() * 1000
        self._build_features(incremental=True, new_data=new_df)

    def run(self):
        if self.feature_matrix is None:
            return
        n = self.feature_matrix.shape[0]
        i = 0 if self.cfg.replay_speed > 0 else n - 1
        bar_seconds = 3 * 60
        while self.state.is_running():
            try:
                if self.cfg.live_mode:
                    time.sleep(60)
                    new_ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe,
                                                          since=int(self.last_timestamp) + 1)
                    self.append_new_data(new_ohlcv)
                    i = self.feature_matrix.shape[0] - 1
                elif self.cfg.replay_speed > 0:
                    sleep_time = bar_seconds / self.cfg.replay_speed
                    time.sleep(sleep_time)
                    i = (i + 1) % n
                else:
                    time.sleep(0.2)
                obs = self.feature_matrix[i]
                self.state.update_latest_obs(obs)
            except Exception as e:
                self.logger.error(f"Data run error: {e}")


class BotStats:
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.win_amounts = []
        self.loss_amounts = []

    def update(self, pnl):
        if pnl > 0:
            self.wins += 1
            self.win_amounts.append(pnl)
        elif pnl < 0:
            self.losses += 1
            self.loss_amounts.append(abs(pnl))
        self.total_profit += pnl

    def winrate(self):
        total_trades = self.wins + self.losses
        return self.wins / total_trades if total_trades > 0 else 0.0

    def rr(self):
        avg_win = np.mean(self.win_amounts) if self.win_amounts else 0.0
        avg_loss = np.mean(self.loss_amounts) if self.loss_amounts else 1e-8
        return avg_win / avg_loss


class Bot:
    def __init__(self, bot_id, model):
        self.id = bot_id
        self.model = model
        self.reset()

    def reset(self):
        self.train_score = 0.0
        self.test_score = 0.0
        self.train_trades = 0
        self.test_trades = 0
        self.personality = "Neutral"
        self.train_stats = BotStats()
        self.test_stats = BotStats()

    def set_personality(self, avg_lev, avg_risk, trades):
        if avg_lev > 5 and trades > 50:
            self.personality = "Aggressive"
        elif avg_risk < 0.3 and trades < 20:
            self.personality = "Conservative"
        else:
            self.personality = "Balanced"


class NewsEffect:
    def __init__(self, cfg, state, logger):
        self.cfg = cfg
        self.state = state
        self.logger = logger
        self.last_payload = None

    def update_payload(self, payload):
        self.last_payload = payload

    def apply(self, obs):
        return obs


class Engine:
    def __init__(self, cfg, state, data, logger):
        self.cfg = cfg
        self.state = state
        self.data = data
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.input_dim = self.data.feature_dim
        self.population_size = self.cfg.population_size
        self.senate_size = self.cfg.senate_size
        self.elite_count = max(2, int(self.population_size * 0.25))
        self.mutation_scale = self.cfg.mutation_scale
        self.max_leverage = self.cfg.max_leverage
        self.train_sleep = self.cfg.train_sleep
        self.bots = []
        self.senate = []
        self.news_effect = NewsEffect(cfg, state, logger)
        self._init_or_load_population()
        self.generation = 0
        self.exchange = ccxt.binance() if self.cfg.real_trade_mode else ccxt.binance(
            {'options': {'defaultType': 'future'}})
        self.optimizers = [optim.Adam(b.model.parameters(), lr=self.cfg.rl_lr) for b in self.bots]

    def _init_or_load_population(self):
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        self.bots = []
        for i in range(self.population_size):
            model_path = os.path.join(model_dir, f"bot_{i}.pth")
            if os.path.exists(model_path):
                model = BotModel(self.input_dim)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                self.logger.info(f"Loaded model for bot {i}")
            else:
                model = BotModel(self.input_dim).to(self.device)
            self.bots.append(Bot(i, model))

    def save_population(self):
        model_dir = "saved_models"
        for b in self.bots:
            model_path = os.path.join(model_dir, f"bot_{b.id}.pth")
            torch.save(b.model.state_dict(), model_path)
        self.logger.info("Saved population models")

    def run(self):
        while self.state.is_running() and self.generation < self.cfg.max_generations:
            try:
                self._run_generation()
                self._update_senate()
                self._update_live_signal()
                self._check_overfitting()
                if self.train_sleep > 0:
                    time.sleep(self.train_sleep)
            except Exception as e:
                self.logger.error(f"Engine error: {e}")
                time.sleep(0.5)
        if self.generation >= self.cfg.max_generations:
            self.logger.info("Max generations reached; ready for real trading")
            self._prepare_real_trade()

    def _run_generation(self):
        for b in self.bots:
            b.reset()
        for b_idx, b in enumerate(self.bots):
            trajectories = []
            for _ in range(self.cfg.rl_epochs):
                obs_train, price_train = self.data.sample_episode("train")
                obs_test, price_test = self.data.sample_episode("test")
                if obs_train is None or obs_test is None:
                    continue
                train_traj = self._collect_trajectory(b, obs_train, price_train)
                test_traj = self._collect_trajectory(b, obs_test, price_test)
                trajectories.append(train_traj)
                b.train_score += train_traj['equity']
                b.test_score += test_traj['equity']
                b.train_trades += train_traj['trades']
                b.test_trades += test_traj['trades']
                self.state.update_equity_curve(b.id, test_traj['curve'])
            self._ppo_update(b_idx, trajectories)
            b.train_score /= self.cfg.rl_epochs
            b.test_score /= self.cfg.rl_epochs
            b.train_trades /= self.cfg.rl_epochs
            b.test_trades /= self.cfg.rl_epochs
            b.set_personality((1 + (self.max_leverage - 1) * 0.5), 0.5, (b.train_trades + b.test_trades) / 2)

        self._evolve()
        self._update_gen_stats()
        self.generation += 1

    def _collect_trajectory(self, bot, obs, price):
        if obs.shape[0] == 0:
            return {'equity': 100.0, 'trades': 0, 'curve': [100.0], 'log_probs': [], 'values': [], 'rewards': []}
        equity = 100.0
        position = 0
        entry_price = 0.0
        trades = 0
        curve = [equity]
        log_probs = []
        values = []
        rewards = []
        t_count = obs.shape[0]
        for i in range(t_count):
            x_np = obs[i]
            x_news = self.news_effect.apply(x_np)
            x = torch.tensor(x_news, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_logits, lever_raw, risk_raw, value = bot.model(x)
            action_probs = F.softmax(action_logits, dim=1)
            action = torch.multinomial(action_probs, 1).item()
            log_prob_action = F.log_softmax(action_logits, dim=1)[0, action]
            lev = 1.0 + torch.sigmoid(lever_raw).item() * (self.max_leverage - 1.0)
            risk_scale = torch.sigmoid(risk_raw).item()
            p = price[i]
            pnl = 0.0
            if position == 0:
                if action != 2:
                    position = 1 if action == 0 else -1
                    entry_price = p
                    trades += 1
            else:
                change = (p - entry_price) / entry_price * position
                pnl = equity * change * lev * risk_scale
                equity += pnl
                if equity <= 0:
                    equity = 0.0
                curve.append(equity)
                if (position > 0 and action == 0) or (position < 0 and action == 1):
                    entry_price = p
                else:
                    position = 0
                    entry_price = 0.0
                    if action != 2:
                        position = 1 if action == 0 else -1
                        entry_price = p
                        trades += 1
            rewards.append(pnl)
            log_probs.append(log_prob_action.item())
            values.append(value.item())
        return {'equity': equity, 'trades': trades, 'curve': curve, 'log_probs': log_probs, 'values': values,
                'rewards': rewards}

    def _ppo_update(self, b_idx, trajectories):
        # Make sure latest_obs is initialized right after features are built
        if self.feature_matrix is not None and len(self.feature_matrix) > 0:
            self.state.update_latest_obs(self.feature_matrix[-1])
        for traj in trajectories:
            rewards = traj['rewards']
            values = traj['values']
            log_probs = traj['log_probs']
            advantages = []
            gae = 0.0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + self.cfg.rl_gamma * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
                gae = delta + self.cfg.rl_gamma * self.cfg.rl_gae_lambda * gae
                advantages.insert(0, gae)
            advantages = torch.tensor(advantages, device=self.device)
            old_log_probs = torch.tensor(log_probs, device=self.device)
            self.optimizers[b_idx].zero_grad()
            for epoch in range(self.cfg.rl_epochs):
                for start in range(0, len(rewards), self.cfg.rl_batch_size):
                    end = start + self.cfg.rl_batch_size
                    batch_adv = advantages[start:end]
                    batch_old_log = old_log_probs[start:end]
                    # Safe fallback – use the last observation from feature_matrix if latest_obs not set yet
                    if self.data.state.get_latest_obs() is None:
                        obs = self.data.feature_matrix[-1]  # last row
                    else:
                        obs = self.data.state.get_latest_obs()

                    batch_x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(
                        end - start, 1)
                    action_logits, lever_raw, risk_raw, value = self.bots[b_idx].model(batch_x)
                    action_probs = F.softmax(action_logits, dim=1)
                    action = torch.multinomial(action_probs, 1).squeeze()
                    new_log_probs = F.log_softmax(action_logits, dim=1).gather(1, action.unsqueeze(1)).squeeze()
                    ratio = torch.exp(new_log_probs - batch_old_log)
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1 - self.cfg.rl_clip_ratio, 1 + self.cfg.rl_clip_ratio) * batch_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(value.squeeze(),
                                             torch.tensor(rewards[start:end], device=self.device) + torch.tensor(
                                                 values[start + 1:end + 1] if start + 1 < len(values) else [0] * (
                                                             end - start), device=self.device))
                    entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(1).mean()
                    loss = actor_loss + self.cfg.rl_value_coef * critic_loss - self.cfg.rl_entropy_coef * entropy
                    loss.backward()
                    self.optimizers[b_idx].step()

    def _evolve(self):
        sorted_bots = sorted(self.bots, key=lambda b: b.test_score, reverse=True)
        elites = sorted_bots[:self.elite_count]
        best_score = elites[0].test_score if elites else 0
        worst_score = sorted_bots[-1].test_score if sorted_bots else 0
        self.state.update_engine_status(self.generation, best_score, worst_score)
        new_bots = []
        for i in range(self.population_size):
            parent = random.choice(elites)
            child_model = self._clone_and_mutate(parent.model)
            new_bots.append(Bot(i, child_model))
        self.bots = new_bots
        self.optimizers = [optim.Adam(b.model.parameters(), lr=self.cfg.rl_lr) for b in self.bots]

    def _clone_and_mutate(self, model):
        child = BotModel(self.input_dim).to(self.device)
        with torch.no_grad():
            for (n, p_child), (n2, p_parent) in zip(child.named_parameters(), model.named_parameters()):
                noise = torch.randn_like(p_parent, device=self.device) * self.mutation_scale
                p_child.copy_(p_parent + noise)
        return child

    def _update_gen_stats(self):
        sorted_bots = sorted(self.bots, key=lambda b: b.test_score, reverse=True)
        best_bot_id = sorted_bots[0].id if sorted_bots else -1
        avg_winrate = np.mean([b.test_stats.winrate() for b in self.bots])
        avg_rr = np.mean([b.test_stats.rr() for b in self.bots])
        avg_profit = np.mean([b.test_stats.total_profit for b in self.bots])
        stats = {'gen': self.generation, 'avg_winrate': avg_winrate, 'avg_rr': avg_rr, 'avg_profit': avg_profit,
                 'best_bot_id': best_bot_id}
        self.state.update_gen_stats(stats)

    def _update_senate(self):
        sorted_bots = sorted(self.bots, key=lambda b: b.test_score, reverse=True)
        self.senate = sorted_bots[:min(self.senate_size, len(sorted_bots))]
        payload = []
        for b in self.senate:
            payload.append({
                "id": b.id,
                "train_score": float(b.train_score),
                "test_score": float(b.test_score),
                "train_trades": int(b.train_trades),
                "test_trades": int(b.test_trades),
                "personality": b.personality,
                "winrate": b.test_stats.winrate(),
                "rr": b.test_stats.rr(),
                "profit": b.test_stats.total_profit
            })
        self.state.update_best_bots(payload)

    def _update_live_signal(self):
        obs = self.state.get_latest_obs()
        if obs is None:
            return
        obs_vec = np.asarray(obs, dtype=np.float32)
        obs_vec = self.news_effect.apply(obs_vec)
        x = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        votes = []
        for b in self.senate:
            with torch.no_grad():
                action_logits, lever_raw, risk_raw, _ = b.model(x)
            probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
            action = np.argmax(probs)
            lev = 1.0 + torch.sigmoid(lever_raw).item() * (self.max_leverage - 1.0)
            conf = float(np.max(probs))
            risk = float(torch.sigmoid(risk_raw).item())
            side = "long" if action == 0 else "short" if action == 1 else "flat"
            votes.append({"bot_id": b.id, "side": side, "confidence": conf, "leverage": lev, "risk": risk})
        if not votes:
            return
        side_scores = {"long": 0.0, "short": 0.0, "flat": 0.0}
        lev_acc = 0.0
        risk_acc = 0.0
        conf_acc = 0.0
        for v in votes:
            side_scores[v["side"]] += v["confidence"]
            lev_acc += v["leverage"] * v["confidence"]
            risk_acc += v["risk"] * v["confidence"]
            conf_acc += v["confidence"]
        final_side = max(side_scores.items(), key=lambda x: x[1])[0]
        avg_lev = lev_acc / conf_acc if conf_acc > 0 else 1.0
        avg_risk = risk_acc / conf_acc if conf_acc > 0 else 0.5
        signal = {"side": final_side, "avg_leverage": float(avg_lev), "avg_risk": float(avg_risk), "votes": votes,
                  "generation": int(self.generation)}
        self.state.update_live_signal(signal)

    def _check_overfitting(self):
        for b in self.bots:
            if b.test_score > 0 and b.train_score / b.test_score > self.cfg.overfit_threshold:
                self.logger.warning(
                    f"Bot {b.id} may be overfitting: train {b.train_score:.2f} vs test {b.test_score:.2f}")

    def _prepare_real_trade(self):
        self.logger.info("Preparing for real trading with best senate")


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_curve(self, curve, title="Equity Curve", xlabel="Bars", ylabel="Equity"):
        self.axes.clear()
        self.axes.plot(curve)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.draw()

    def plot_bar(self, data, labels, title="Bot Scores", xlabel="Bots", ylabel="Score"):
        self.axes.clear()
        self.axes.bar(labels, data)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.draw()

    def plot_multi_line(self, data_dict, title="Generation Metrics", xlabel="Generations", ylabel="Value"):
        self.axes.clear()
        for label, values in data_dict.items():
            self.axes.plot(values, label=label)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.legend()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self, cfg, state, engine, data, logger):
        super().__init__()
        self.cfg = cfg
        self.state = state
        self.engine = engine
        self.data = data
        self.logger = logger
        self.setWindowTitle("Quant Senate Workstation")
        self.resize(1600, 900)
        self._build_ui()
        self.t_engine = None
        self.t_data = None
        self.start_threads()
        self.statusBar().showMessage("Ready")
        self._apply_styles()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; font-size: 14px; }
            QPushButton { background-color: #007bff; color: #ffffff; border: none; padding: 8px 16px; border-radius: 4px; font-size: 14px; }
            QPushButton:hover { background-color: #0056b3; }
            QTextEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #444; font-size: 13px; }
            QComboBox { background-color: #2d2d2d; color: #ffffff; border: 1px solid #444; padding: 5px; font-size: 13px; }
            QComboBox QAbstractItemView { background-color: #2d2d2d; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #444; background-color: #1e1e1e; }
            QTabBar::tab { background-color: #2d2d2d; color: #ffffff; padding: 10px; border: 1px solid #444; }
            QTabBar::tab:selected { background-color: #007bff; }
            QProgressBar { background-color: #2d2d2d; color: #ffffff; border: 1px solid #444; text-align: center; }
            QProgressBar::chunk { background-color: #007bff; }
            QLineEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #444; padding: 5px; font-size: 13px; }
            QFrame { border: 1px solid #444; }
        """)

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        self.tab_widget.addTab(self._build_chart_panel(), "Charts")
        self.tab_widget.addTab(self._build_engine_panel(), "Engine")
        self.tab_widget.addTab(self._build_signal_panel(), "Signals")
        self.tab_widget.addTab(self._build_analytics_panel(), "Analytics")
        self.tab_widget.addTab(self._build_gen_metrics_panel(), "Gen Metrics")
        self.tab_widget.addTab(self._build_log_panel(), "Logs")
        self.tab_widget.addTab(self._build_settings_panel(), "Settings")

    def _build_chart_panel(self):
        w = QFrame()
        splitter = QSplitter(Qt.Horizontal)
        w.setLayout(QVBoxLayout())
        w.layout().addWidget(splitter)

        tv_frame = QFrame()
        tv_layout = QVBoxLayout()
        tv_frame.setLayout(tv_layout)
        self.web = QWebEngineView()
        html = self._tradingview_html(self.cfg.symbol, self.cfg.timeframe)
        self.web.setHtml(html)
        tv_layout.addWidget(self.web)

        price_frame = QFrame()
        price_layout = QVBoxLayout()
        price_frame.setLayout(price_layout)
        self.price_canvas = PlotCanvas(self)
        price_layout.addWidget(self.price_canvas)

        splitter.addWidget(tv_frame)
        splitter.addWidget(price_frame)
        splitter.setSizes([700, 700])
        return w

    def _build_engine_panel(self):
        w = QFrame()
        layout = QVBoxLayout()
        w.setLayout(layout)
        self.engine_status_label = QLabel("Gen: 0 | Best: 0 | Worst: 0")
        layout.addWidget(self.engine_status_label)
        hbox = QHBoxLayout()
        self.start_engine_btn = QPushButton("Start Engine")
        self.stop_engine_btn = QPushButton("Stop Engine")
        self.start_engine_btn.clicked.connect(self.start_threads)
        self.stop_engine_btn.clicked.connect(self.stop_threads)
        hbox.addWidget(self.start_engine_btn)
        hbox.addWidget(self.stop_engine_btn)
        layout.addLayout(hbox)
        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, self.cfg.max_generations)
        layout.addWidget(self.gen_progress)
        self.bot_selector = QComboBox()
        self.bot_selector.currentIndexChanged.connect(self.on_bot_selected)
        layout.addWidget(self.bot_selector)
        self.best_bots_text = QTextEdit()
        self.best_bots_text.setReadOnly(True)
        layout.addWidget(self.best_bots_text, 1)
        self.plot_canvas = PlotCanvas(self)
        layout.addWidget(self.plot_canvas, 1)
        return w

    def _build_signal_panel(self):
        w = QFrame()
        layout = QVBoxLayout()
        w.setLayout(layout)
        self.signal_label = QLabel("No signal")
        layout.addWidget(self.signal_label)
        btn_row = QHBoxLayout()
        self.approve_btn = QPushButton("Approve")
        self.reject_btn = QPushButton("Reject")
        self.approve_btn.clicked.connect(self.on_approve)
        self.reject_btn.clicked.connect(self.on_reject)
        btn_row.addWidget(self.approve_btn)
        btn_row.addWidget(self.reject_btn)
        layout.addLayout(btn_row)
        self.debate_text = QTextEdit()
        self.debate_text.setReadOnly(True)
        layout.addWidget(self.debate_text, 1)
        return w

    def _build_analytics_panel(self):
        w = QFrame()
        layout = QVBoxLayout()
        w.setLayout(layout)
        self.analytics_canvas = PlotCanvas(self)
        layout.addWidget(self.analytics_canvas, 1)
        update_btn = QPushButton("Update Analytics")
        update_btn.clicked.connect(self.update_analytics)
        layout.addWidget(update_btn)
        return w

    def _build_gen_metrics_panel(self):
        w = QFrame()
        layout = QVBoxLayout()
        w.setLayout(layout)
        self.gen_metrics_canvas = PlotCanvas(self)
        layout.addWidget(self.gen_metrics_canvas, 1)
        update_btn = QPushButton("Update Gen Metrics")
        update_btn.clicked.connect(self.update_gen_metrics)
        layout.addWidget(update_btn)
        return w

    def _build_log_panel(self):
        w = QFrame()
        layout = QVBoxLayout()
        w.setLayout(layout)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, 1)
        return w

    def _build_settings_panel(self):
        w = QFrame()
        layout = QFormLayout()
        w.setLayout(layout)
        self.symbol_edit = QLineEdit(self.cfg.symbol)
        layout.addRow("Symbol:", self.symbol_edit)
        self.timeframe_edit = QLineEdit(self.cfg.timeframe)
        layout.addRow("Timeframe:", self.timeframe_edit)
        self.pop_size_edit = QLineEdit(str(self.cfg.population_size))
        layout.addRow("Population Size:", self.pop_size_edit)
        self.real_trade_toggle = QComboBox()
        self.real_trade_toggle.addItems(["Paper", "Real"])
        layout.addRow("Trade Mode:", self.real_trade_toggle)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        return w

    def _tradingview_html(self, symbol, interval):
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Live Chart</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{ margin:0; background:#050710; }}
        #chart {{ width:100%; height:100vh; }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
            width: window.innerWidth,
            height: window.innerHeight,
            layout: {{ background: {{ color: '#050710' }}, textColor: '#d1d4dc' }},
            grid: {{ vertLines: {{ color: '#2d2d2d' }}, horzLines: {{ color: '#2d2d2d' }} }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{ borderColor: '#2d2d2d' }},
            timeScale: {{ borderColor: '#2d2d2d', timeVisible: true, secondsVisible: false }},
        }});

        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#00c4b4',
            downColor: '#ff5252',
            borderVisible: false,
            wickUpColor: '#00c4b4',
            wickDownColor: '#ff5252',
        }});

        async function loadData() {{
            const response = await fetch(`https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000`);
            const data = await response.json();
            const candlesticks = data.map(d => ({{
                time: d[0] / 1000,
                open: parseFloat(d[1]),
                high: parseFloat(d[2]),
                low: parseFloat(d[3]),
                close: parseFloat(d[4])
            }}));
            candleSeries.setData(candlesticks);

            setInterval(async () => {{
                const latest = await fetch(`https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1`).then(r => r.json());
                const newCandle = {{
                    time: latest[0][0] / 1000,
                    open: parseFloat(latest[0][1]),
                    high: parseFloat(latest[0][2]),
                    low: parseFloat(latest[0][3]),
                    close: parseFloat(latest[0][4])
                }};
                candleSeries.update(newCandle);
            }}, 10000);
        }}

        loadData();
        window.onresize = () => chart.applyOptions({{ width: window.innerWidth, height: window.innerHeight }});
    </script>
</body>
</html>
"""

    def refresh(self):
        try:
            status = self.state.get_engine_status()
            if status is not None:
                gen, best, worst = status
                self.engine_status_label.setText(
                    f"Gen: {gen} / {self.cfg.max_generations} | Best: {best:.2f} | Worst: {worst:.2f}")
                self.gen_progress.setValue(gen)
            best_bots = self.state.get_best_bots()
            self.bot_selector.blockSignals(True)
            self.bot_selector.clear()
            if best_bots:
                lines = []
                for b in best_bots:
                    self.bot_selector.addItem(
                        f"Bot {b['id']} ({b['personality']}) Winrate: {b['winrate']:.2f} R:R: {b['rr']:.2f} Profit: {b['profit']:.2f}")
                    line = f"id={b['id']} personality={b['personality']} train_eq={b['train_score']:.2f} test_eq={b['test_score']:.2f} trades_train={b['train_trades']} trades_test={b['test_trades']} winrate={b['winrate']:.2f} rr={b['rr']:.2f} profit={b['profit']:.2f}"
                    lines.append(line)
                self.best_bots_text.setPlainText("\n".join(lines))
            self.bot_selector.blockSignals(False)
            signal = self.state.get_live_signal()
            if signal is not None:
                side = signal.get("side", "-")
                lev = signal.get("avg_leverage", 1.0)
                risk = signal.get("avg_risk", 0.5)
                gen = signal.get("generation", 0)
                self.signal_label.setText(f"Signal: {side} | lev={lev:.2f} | risk={risk:.2f} | gen={gen}")
                votes = signal.get("votes", [])
                lines = []
                for v in votes:
                    line = f"Bot {v['bot_id']}: {v['side']} conf={v['confidence']:.2f} lev={v['leverage']:.2f} risk={v['risk']:.2f}"
                    lines.append(line)
                self.debate_text.setPlainText("\n".join(lines))
            else:
                self.signal_label.setText("No signal")
                self.debate_text.setPlainText("")
            prices = self.state.get_price_history()
            if prices:
                self.price_canvas.plot_curve(prices, title="Price History", xlabel="Time", ylabel="Price")
        except Exception as e:
            self.logger.error(f"Refresh error: {e}")
            QMessageBox.warning(self, "Error", str(e))

    def update_analytics(self):
        best_bots = self.state.get_best_bots()
        if best_bots:
            scores = [b['test_score'] for b in best_bots]
            labels = [f"Bot {b['id']}" for b in best_bots]
            self.analytics_canvas.plot_bar(scores, labels, title="Top Bot Test Scores", xlabel="Bots",
                                           ylabel="Test Score")

    def update_gen_metrics(self):
        gen_stats = self.state.get_gen_stats()
        if gen_stats:
            gens = [s['gen'] for s in gen_stats]
            data_dict = {
                'Avg Winrate': [s['avg_winrate'] for s in gen_stats],
                'Avg R:R': [s['avg_rr'] for s in gen_stats],
                'Avg Profit': [s['avg_profit'] for s in gen_stats]
            }
            self.gen_metrics_canvas.plot_multi_line(data_dict, title="Generation Metrics Over Time",
                                                    xlabel="Generation", ylabel="Value")

    def on_bot_selected(self, index):
        if index < 0:
            return
        bot_id = self.state.get_best_bots()[index]["id"]
        curve = self.state.get_equity_curve(bot_id)
        if curve:
            self.plot_canvas.plot_curve(curve)

    def on_approve(self):
        signal = self.state.get_live_signal()
        if signal is None:
            return
        payload = {"ts": time.time(), "decision": "approve", "signal": signal}
        self.state.update_last_action(payload)
        with open("signals.csv", "a") as f:
            f.write(f"{payload['ts']},{payload['decision']},{signal['side']}\n")

    def on_reject(self):
        signal = self.state.get_live_signal()
        if signal is None:
            return
        payload = {"ts": time.time(), "decision": "reject", "signal": signal}
        self.state.update_last_action(payload)

    def save_settings(self):
        try:
            old_symbol = self.cfg.symbol
            self.cfg.symbol = self.symbol_edit.text()
            self.cfg.timeframe = self.timeframe_edit.text()
            self.cfg.population_size = int(self.pop_size_edit.text())
            self.cfg.real_trade_mode = self.real_trade_toggle.currentText() == "Real"
            if self.cfg.symbol != old_symbol:
                self.data.symbol = self.cfg.symbol.replace('/', '')
                self.data._load_or_fetch()
            html = self._tradingview_html(self.cfg.symbol, self.cfg.timeframe)
            self.web.setHtml(html)
            self.statusBar().showMessage("Settings saved")
            self.stop_threads()
            self.engine = Engine(self.cfg, self.state, self.data, self.logger)
            self.start_threads()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def start_threads(self):
        if self.t_engine is None or not self.t_engine.is_alive():
            self.t_engine = threading.Thread(target=self.engine.run, daemon=True)
            self.t_engine.start()
        if self.t_data is None or not self.t_data.is_alive():
            self.t_data = threading.Thread(target=self.data.run, daemon=True)
            self.t_data.start()
        self.statusBar().showMessage("Threads started")

    def stop_threads(self):
        self.state.stop()
        if self.t_engine:
            self.t_engine.join(timeout=1)
        if self.t_data:
            self.t_data.join(timeout=1)
        self.state.running = True
        self.statusBar().showMessage("Threads stopped")

    def closeEvent(self, event):
        self.stop_threads()
        self.engine.save_population()
        event.accept()


def main():
    cfg = AppConfig()
    app = QApplication(sys.argv)
    log_widget = QTextEdit()
    logger = create_logger(cfg, log_widget=log_widget)
    state = SharedState(cfg, logger)
    data = DataManager(cfg, state, logger)
    if data.df is None:
        logger.error("Failed to initialize data; exiting")
        sys.exit(1)
    engine = Engine(cfg, state, data, logger)
    window = MainWindow(cfg, state, engine, data, logger)
    window.log_text = log_widget
    timer = QTimer()
    timer.timeout.connect(window.refresh)
    timer.start(50)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
