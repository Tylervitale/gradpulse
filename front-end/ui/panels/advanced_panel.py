from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QPushButton,
    QFormLayout, QGroupBox, QSpinBox, QHBoxLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

from core.worker import Worker
from gradpulse import rl
from gradpulse.compression import compress_rle, compress_delta, compress_spline, verify_compression

from PyQt6.QtWidgets import QComboBox, QLineEdit, QListWidget, QTextEdit, QSplitter
from gradpulse.dlp import Rule, Proposition, SoftLogic, SoftRelational
from gradpulse.scheduling import DependencyGraph, OperationNode
from gradpulse.microscheduler import Microscheduler
from gradpulse.distortion import Predistorter
from gradpulse.benchmark import run_benchmark
from ui.components.mpl_widget import MatplotlibWidget


class AdvancedPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.threadpool = self.main_window.get_threadpool()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        # 1. RL Agent Tab
        self.rl_tab = QWidget()
        self.init_rl_tab()
        self.tabs.addTab(self.rl_tab, "Reinforcement Learning")

        # 2. Compression Tab
        self.compress_tab = QWidget()
        self.init_compress_tab()
        self.tabs.addTab(self.compress_tab, "Waveform Compression")

        # 3. Error Mitigation (ZNE) Tab
        self.zne_tab = QWidget()
        self.init_zne_tab()
        self.tabs.addTab(self.zne_tab, "Error Mitigation (ZNE)")



        # 4. DLP Tab
        self.dlp_tab = QWidget()
        self.init_dlp_tab()
        self.tabs.addTab(self.dlp_tab, "Differentiable Logic (DLP)")

        # 5. Scheduling Tab
        self.scheduling_tab = QWidget()
        self.init_scheduling_tab()
        self.tabs.addTab(self.scheduling_tab, "Advanced Scheduling")

        # 6. Pre-Distortion Tab
        self.distortion_tab = QWidget()
        self.init_distortion_tab()
        self.tabs.addTab(self.distortion_tab, "Cable Pre-Distortion")

        # 7. Benchmarking Tab
        self.benchmark_tab = QWidget()
        self.init_benchmark_tab()
        self.tabs.addTab(self.benchmark_tab, "Benchmarking")

        # 8. MPS Evaluator Tab
        self.mps_tab = QWidget()
        self.init_mps_tab()
        self.tabs.addTab(self.mps_tab, "MPS Evaluator")

        layout.addWidget(self.tabs)

    def init_rl_tab(self):
        layout = QVBoxLayout(self.rl_tab)

        group = QGroupBox("Train PPO Agent (Stable-Baselines3)")
        form = QFormLayout()

        self.rl_steps = QSpinBox()
        self.rl_steps.setRange(100, 100000)
        self.rl_steps.setValue(2048)
        form.addRow("Total Timesteps:", self.rl_steps)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.run_rl_training)
        form.addRow("", self.train_btn)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def init_compress_tab(self):
        layout = QVBoxLayout(self.compress_tab)

        group = QGroupBox("Waveform Compression")
        vbox = QVBoxLayout()

        self.compress_method_combo = QComboBox()
        self.compress_method_combo.addItems(["RLE", "Delta", "Spline"])
        vbox.addWidget(QLabel("Compression Method:"))
        vbox.addWidget(self.compress_method_combo)

        self.compress_btn = QPushButton("Compress Current Optimized Pulse")
        self.compress_btn.clicked.connect(self.run_compression)
        vbox.addWidget(self.compress_btn)

        self.compress_results = QLabel("Ready.")
        vbox.addWidget(self.compress_results)

        group.setLayout(vbox)
        layout.addWidget(group)
        layout.addStretch()

    def init_zne_tab(self):
        layout = QVBoxLayout(self.zne_tab)

        group = QGroupBox("Zero-Noise Extrapolation Configuration")
        form = QFormLayout()

        self.zne_method_combo = QComboBox()
        self.zne_method_combo.addItems(["stretch", "fold"])
        form.addRow("Scaling Method:", self.zne_method_combo)

        self.zne_scale = QDoubleSpinBox()
        self.zne_scale.setRange(1.0, 5.0)
        self.zne_scale.setValue(1.5)
        self.zne_scale.setSingleStep(0.1)
        form.addRow("Noise Scale Factor:", self.zne_scale)

        self.zne_btn = QPushButton("Run ZNE Pipeline")
        self.zne_btn.clicked.connect(self.run_zne)
        form.addRow("", self.zne_btn)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def run_rl_training(self):
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training...")

        steps = self.rl_steps.value()

        def train_task():
            # Mock or call actual gradpulse.rl implementation depending on import
            print(f"Starting RL training for {steps} steps on CrossResonanceEnv...")
            return rl.train_ppo(total_timesteps=steps)

        worker = Worker(train_task)
        worker.signals.result.connect(self.on_rl_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.train_btn, "Start Training"))

        self.threadpool.start(worker)

    def run_compression(self):
        opt_panel = self.main_window.opt_panel
        if not opt_panel.result or 'best_waveform' not in opt_panel.result:
            self.compress_results.setText("Error: No pulse found. Run optimization first.")
            return

        pulse = opt_panel.result['best_waveform']

        def compress_task():
            import numpy as np
            # Convert torch tensor if needed to numpy
            arr = pulse.numpy() if hasattr(pulse, "numpy") else np.array(pulse)
            method = self.compress_method_combo.currentText()

            if method == "RLE":
                compressed = compress_rle(arr)
            elif method == "Delta":
                compressed = compress_delta(arr)
            elif method == "Spline":
                compressed = compress_spline(arr)

            valid = verify_compression(arr, compressed)
            return compressed, valid, arr.nbytes

        worker = Worker(compress_task)
        worker.signals.result.connect(self.on_compress_success)
        worker.signals.error.connect(self.on_error)

        self.threadpool.start(worker)

    def run_zne(self):
        self.zne_btn.setEnabled(False)
        print(f"Starting ZNE scaling with factor {self.zne_scale.value()}...")
        scale = self.zne_scale.value()

        opt_panel = self.main_window.opt_panel
        if not opt_panel.result or 'best_waveform' not in opt_panel.result:
            print("Error: No pulse found. Run optimization first.")
            self._reset_btn(self.zne_btn, "Run ZNE Pipeline")
            return

        pulse = opt_panel.result['best_waveform']

        method = self.zne_method_combo.currentText()
        def zne_task():
            from gradpulse.mitigation import stretch_pulse, fold_pulse
            import numpy as np
            arr = pulse.numpy() if hasattr(pulse, "numpy") else np.array(pulse)
            if method == "fold":
                scaled = fold_pulse(arr, scale)
            else:
                scaled = stretch_pulse(arr, scale)
            return scaled

        worker = Worker(zne_task)
        worker.signals.result.connect(self.on_zne_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.zne_btn, "Run ZNE Pipeline"))

        self.threadpool.start(worker)

    def on_zne_success(self, scaled):
        print(f"ZNE scaled pulse size: {scaled.shape}")

    def on_rl_success(self, model):
        print("RL Training Completed Successfully.")

    def on_compress_success(self, result):
        compressed, is_valid, orig_size = result
        new_size = sum(v.nbytes if hasattr(v, "nbytes") else 0 for v in compressed.values()) if isinstance(compressed, dict) else len(compressed)*8

        self.compress_results.setText(
            f"Compression Complete:\n"
            f"Original Size: {orig_size} bytes\n"
            f"Compressed Size: ~{new_size} bytes\n"
            f"Lossless Verification: {'PASS' if is_valid else 'FAIL'}"
        )
        print("Compression algorithm ran.")

    def on_error(self, error):
        print(f"Error in Advanced Features: {error[1]}")

    def _reset_btn(self, btn, text):
        btn.setEnabled(True)
        btn.setText(text)

    def init_dlp_tab(self):
        layout = QVBoxLayout(self.dlp_tab)

        group = QGroupBox("DLP Rule Configuration")
        form = QFormLayout()

        self.dlp_prop_name = QLineEdit()
        self.dlp_prop_name.setPlaceholderText("e.g. pulse_amplitude_limit")
        form.addRow("Proposition Name:", self.dlp_prop_name)

        self.dlp_penalty = QDoubleSpinBox()
        self.dlp_penalty.setRange(0.1, 100.0)
        self.dlp_penalty.setValue(1.0)
        form.addRow("Penalty Weight:", self.dlp_penalty)

        self.add_rule_btn = QPushButton("Add Rule")
        self.add_rule_btn.clicked.connect(self.add_dlp_rule)
        form.addRow("", self.add_rule_btn)

        self.rules_list = QListWidget()
        form.addRow("Active Rules:", self.rules_list)

        self.run_dlp_btn = QPushButton("Apply Rules to Optimization")
        self.run_dlp_btn.clicked.connect(self.run_dlp_optimization)
        form.addRow("", self.run_dlp_btn)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def add_dlp_rule(self):
        name = self.dlp_prop_name.text()
        weight = self.dlp_penalty.value()
        if name:
            self.rules_list.addItem(f"{name} (Weight: {weight})")
            self.dlp_prop_name.clear()

    def run_dlp_optimization(self):
        self.run_dlp_btn.setEnabled(False)
        self.run_dlp_btn.setText("Running...")

        rules = []
        for i in range(self.rules_list.count()):
            item_text = self.rules_list.item(i).text()
            name = item_text.split(" (Weight: ")[0]
            weight = float(item_text.split(" (Weight: ")[1].strip(")"))
            rules.append((name, weight))

        print(f"Applying DLP rules: {rules}")

        def dlp_task():
            import torch
            from gradpulse.dlp import SoftRelational, Rule
            import numpy as np

            # We build some real Rule instances to demonstrate applying them
            # For this UI, we can evaluate a dummy metrics dict against the created rules
            metrics = {'fidelity': torch.tensor(0.99), 'leakage': torch.tensor(0.001)}
            rule_objects = []

            for rule_name, weight in rules:
                # We map some common strings to real proposition evaluation
                if "leakage" in rule_name.lower():
                     cond = lambda m: SoftRelational.greater_than(m.get('leakage', torch.tensor(0.0)), threshold=0.005)
                     cons = lambda m: torch.tensor(0.0) # Consequence is false (we don't want this)
                     rule_objects.append(Rule(cond, cons, weight=weight))
                else:
                     # Generic rule
                     cond = lambda m: torch.tensor(1.0) # always true
                     cons = lambda m: torch.tensor(1.0) # always true
                     rule_objects.append(Rule(cond, cons, weight=weight))

            total_penalty = 0.0
            for r in rule_objects:
                 total_penalty += r.evaluate(metrics).item()

            return f"Applied {len(rule_objects)} rules to test metrics. Total Penalty: {total_penalty:.5f}"

        worker = Worker(dlp_task)
        worker.signals.result.connect(self.on_dlp_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_dlp_btn, "Apply Rules to Optimization"))
        self.threadpool.start(worker)

    def on_dlp_success(self, result):
        print(f"DLP success: {result}")

    def init_scheduling_tab(self):
        layout = QVBoxLayout(self.scheduling_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        group = QGroupBox("Construct Schedule Graph")
        form = QFormLayout()

        self.sched_node_name = QLineEdit()
        self.sched_node_name.setPlaceholderText("e.g. X_0, CZ_01")
        form.addRow("Node Name:", self.sched_node_name)

        self.sched_duration = QSpinBox()
        self.sched_duration.setRange(10, 1000)
        self.sched_duration.setValue(50)
        form.addRow("Duration (ns):", self.sched_duration)

        self.sched_channels = QLineEdit()
        self.sched_channels.setPlaceholderText("e.g. q0_drive, q1_drive")
        form.addRow("Channels (csv):", self.sched_channels)

        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self.add_sched_node)
        form.addRow("", self.add_node_btn)

        self.sched_dep_parent = QComboBox()
        self.sched_dep_child = QComboBox()
        form.addRow("Parent Node:", self.sched_dep_parent)
        form.addRow("Child Node:", self.sched_dep_child)

        self.add_dep_btn = QPushButton("Add Dependency")
        self.add_dep_btn.clicked.connect(self.add_sched_dep)
        form.addRow("", self.add_dep_btn)

        self.load_sched_btn = QPushButton("Load Sequence from JSON")
        self.load_sched_btn.clicked.connect(self.load_sched_sequence)
        form.addRow("", self.load_sched_btn)

        self.run_sched_btn = QPushButton("Run MicroScheduler")
        self.run_sched_btn.clicked.connect(self.run_microscheduler)
        form.addRow("", self.run_sched_btn)

        group.setLayout(form)
        control_layout.addWidget(group)

        self.sched_output = QTextEdit()
        self.sched_output.setReadOnly(True)
        control_layout.addWidget(self.sched_output)

        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        self.sched_plot = MatplotlibWidget()
        viz_layout.addWidget(QLabel("Schedule Timeline"))
        viz_layout.addWidget(self.sched_plot)

        splitter.addWidget(control_widget)
        splitter.addWidget(viz_widget)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

        self.sched_nodes = {}
        self.sched_deps = []

    def add_sched_node(self):
        name = self.sched_node_name.text()
        dur = self.sched_duration.value()
        channels = [c.strip() for c in self.sched_channels.text().split(',')] if self.sched_channels.text() else []
        if name and name not in self.sched_nodes:
            self.sched_nodes[name] = {'duration': dur, 'channels': channels}
            self.sched_dep_parent.addItem(name)
            self.sched_dep_child.addItem(name)
            self.sched_node_name.clear()
            self.sched_channels.clear()
            self.update_sched_output()

    def load_sched_sequence(self):
        from PyQt6.QtWidgets import QFileDialog
        import json
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Sequence JSON", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    seq_data = json.load(f)

                self.sched_nodes.clear()
                self.sched_deps.clear()
                self.sched_dep_parent.clear()
                self.sched_dep_child.clear()

                for node in seq_data.get('nodes', []):
                    name = node['name']
                    self.sched_nodes[name] = {'duration': node['duration'], 'channels': node.get('channels', [])}
                    self.sched_dep_parent.addItem(name)
                    self.sched_dep_child.addItem(name)

                for dep in seq_data.get('dependencies', []):
                    self.sched_deps.append((dep['from'], dep['to']))

                self.update_sched_output()
                print(f"Loaded sequence from {file_name}")
            except Exception as e:
                print(f"Failed to load sequence: {e}")

    def add_sched_dep(self):
        parent = self.sched_dep_parent.currentText()
        child = self.sched_dep_child.currentText()
        if parent and child and parent != child:
            dep = (parent, child)
            if dep not in self.sched_deps:
                self.sched_deps.append(dep)
                self.update_sched_output()

    def update_sched_output(self):
        text = "Nodes:\n"
        for n, data in self.sched_nodes.items():
            text += f" - {n} ({data['duration']}ns, channels={data['channels']})\n"
        text += "\nDependencies:\n"
        for p, c in self.sched_deps:
            text += f" - {p} -> {c}\n"
        self.sched_output.setPlainText(text)

    def run_microscheduler(self):
        self.run_sched_btn.setEnabled(False)

        def sched_task():
            from gradpulse.microscheduler import Microscheduler
            from gradpulse.scheduling import DependencyGraph, OperationNode

            graph = DependencyGraph()
            ops = {}
            for name, data in self.sched_nodes.items():
                node = OperationNode(op_id=name, duration_ns=data['duration'], channels=data['channels'], qubits=[])
                ops[name] = node
                graph.add_node(node)

            for p, c in self.sched_deps:
                graph.add_edge(from_id=p, to_id=c)

            scheduler = Microscheduler(dt_ns=1.0)
            schedule = scheduler.schedule(graph)

            # Map node IDs in schedule to actual OperationNode objects for plotting
            result_schedule = {}
            for op_id, start_time in schedule.items():
                result_schedule[ops[op_id]] = start_time

            return result_schedule

        worker = Worker(sched_task)
        worker.signals.result.connect(self.on_sched_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_sched_btn, "Run MicroScheduler"))
        self.threadpool.start(worker)

    def on_sched_success(self, schedule):
        text = self.sched_output.toPlainText()
        text += "\n\nScheduled Times:\n"
        for node, start_time in schedule.items():
            text += f" - {node.op_id}: start={start_time}ns, end={start_time + node.duration_ns}ns\n"
        self.sched_output.setPlainText(text)

        # Plotting Timeline
        self.sched_plot.clear()
        ax = self.sched_plot.get_axes()

        y_ticks = []
        y_labels = []

        for i, (node, start_time) in enumerate(schedule.items()):
            end_time = start_time + node.duration_ns
            ax.barh(i, end_time - start_time, left=start_time, height=0.5, color='skyblue', edgecolor='black')
            ax.text(start_time + (end_time - start_time)/2, i, node.op_id, ha='center', va='center', color='black')
            y_ticks.append(i)
            y_labels.append(node.op_id)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (ns)")
        ax.set_title("MicroScheduler Timeline")
        self.sched_plot.get_canvas().draw()
        print("MicroScheduler run successfully.")

    def init_distortion_tab(self):
        layout = QVBoxLayout(self.distortion_tab)
        group = QGroupBox("Iterative Tikhonov Pre-Distortion")
        form = QFormLayout()

        self.dist_kernel = QLineEdit()
        self.dist_kernel.setPlaceholderText("e.g. 1.0, -0.1, 0.05")
        form.addRow("Transfer Kernel (csv):", self.dist_kernel)

        self.run_dist_btn = QPushButton("Apply Predistorter")
        self.run_dist_btn.clicked.connect(self.run_distortion)
        form.addRow("", self.run_dist_btn)

        group.setLayout(form)
        layout.addWidget(group)

        self.dist_output = QTextEdit()
        self.dist_output.setReadOnly(True)
        layout.addWidget(self.dist_output)

    def run_distortion(self):
        self.run_dist_btn.setEnabled(False)
        opt_panel = self.main_window.opt_panel
        if not opt_panel.result or 'best_waveform' not in opt_panel.result:
            self.dist_output.setPlainText("Error: No optimized pulse found to distort.")
            self._reset_btn(self.run_dist_btn, "Apply Predistorter")
            return

        pulse = opt_panel.result['best_waveform']
        kernel_str = self.dist_kernel.text()

        def dist_task():
            import torch
            import numpy as np
            from gradpulse.distortion import Predistorter

            arr = pulse.numpy() if hasattr(pulse, "numpy") else np.array(pulse)
            try:
                if kernel_str:
                    kernel = np.array([float(x.strip()) for x in kernel_str.split(',')])
                else:
                    # dummy kernel
                    kernel = np.array([1.0, -0.2, 0.05, -0.01])
            except:
                kernel = np.array([1.0, -0.2, 0.05, -0.01])

            distorter = Predistorter(torch.tensor(kernel, dtype=torch.float32))
            # predistorter takes and returns tensors
            distorted_pulse = distorter.predistort(torch.tensor(arr, dtype=torch.float32))
            return distorted_pulse, kernel

        worker = Worker(dist_task)
        worker.signals.result.connect(self.on_dist_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_dist_btn, "Apply Predistorter"))
        self.threadpool.start(worker)

    def on_dist_success(self, result):
        distorted_pulse, kernel = result
        self.dist_output.setPlainText(f"Predistortion applied successfully.\nKernel used: {kernel}\nDistorted Pulse shape: {distorted_pulse.shape}")

    def init_benchmark_tab(self):
        layout = QVBoxLayout(self.benchmark_tab)
        group = QGroupBox("Benchmark: gradpulse vs qutip-qtrl")
        form = QFormLayout()

        self.bench_gate = QComboBox()
        self.bench_gate.addItems(["cnot", "iswap", "cz"])
        form.addRow("Target Gate:", self.bench_gate)

        self.run_bench_btn = QPushButton("Run Benchmark")
        self.run_bench_btn.clicked.connect(self.run_bench)
        form.addRow("", self.run_bench_btn)

        group.setLayout(form)
        layout.addWidget(group)

        self.bench_output = QTextEdit()
        self.bench_output.setReadOnly(True)
        self.bench_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")
        layout.addWidget(self.bench_output)

    def init_mps_tab(self):
        layout = QVBoxLayout(self.mps_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        group = QGroupBox("MPS Witness Evaluator")
        form = QFormLayout()

        self.mps_chi_max = QSpinBox()
        self.mps_chi_max.setRange(4, 256)
        self.mps_chi_max.setValue(64)
        form.addRow("Max Bond Dimension (chi):", self.mps_chi_max)

        self.mps_n_traj = QSpinBox()
        self.mps_n_traj.setRange(10, 1000)
        self.mps_n_traj.setValue(200)
        form.addRow("Trajectories:", self.mps_n_traj)

        self.run_mps_btn = QPushButton("Run MPS Witness")
        self.run_mps_btn.clicked.connect(self.run_mps)
        form.addRow("", self.run_mps_btn)

        group.setLayout(form)
        control_layout.addWidget(group)

        self.mps_output = QTextEdit()
        self.mps_output.setReadOnly(True)
        self.mps_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")

        splitter.addWidget(control_widget)
        splitter.addWidget(self.mps_output)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

    def run_bench(self):
        self.run_bench_btn.setEnabled(False)
        gate = self.bench_gate.currentText()

        def bench_task():
            from gradpulse.benchmark import run_benchmark
            # Max iterations kept small for UI
            res = run_benchmark(gate=gate, max_iter=20, verbose=False)
            return res

        worker = Worker(bench_task)
        worker.signals.result.connect(self.on_bench_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_bench_btn, "Run Benchmark"))
        self.threadpool.start(worker)

    def on_bench_success(self, res):
        text = "--- Benchmark Results ---\n"
        if "gradpulse" in res:
            gp = res["gradpulse"]
            text += f"gradpulse:\n  Fidelity: {gp.get('fidelity', 'N/A')}\n  Wall time: {gp.get('wall_s', 'N/A')} s\n  Iters: {gp.get('iters', 'N/A')}\n\n"
        if "qutip_qtrl" in res:
            qt = res["qutip_qtrl"]
            text += f"qutip-qtrl:\n  Fidelity: {qt.get('fidelity', 'N/A')}\n  Wall time: {qt.get('wall_s', 'N/A')} s\n  Iters: {qt.get('iters', 'N/A')}\n"
        self.bench_output.setPlainText(text)

    def run_mps(self):
        self.run_mps_btn.setEnabled(False)
        self.run_mps_btn.setText("Running...")
        self.mps_output.clear()

        chi_max = self.mps_chi_max.value()
        n_traj = self.mps_n_traj.value()

        def mps_task():
            opt_panel = self.main_window.opt_panel
            if not opt_panel.result or 'best_waveform' not in opt_panel.result:
                raise ValueError("No active optimization result found. Please run an optimization first.")
            result = opt_panel.result
            if 'optimizer' not in result:
                raise ValueError("Optimization result missing 'optimizer' key.")

            opt = result['optimizer']
            pulse = result['best_waveform']

            from gradpulse.mps import ChainTEBD
            import numpy as np
            import itertools

            # Assuming we are running on MultiQubitOptimizer with open system
            if not hasattr(opt, 'profile') or not hasattr(opt.profile, 'n_qubits'):
                raise ValueError("MPS evaluator requires a MultiQubitOptimizer with defined n_qubits.")

            n_qubits = opt.profile.n_qubits

            # Simple ensemble: computational basis states
            ensemble = list(itertools.product([0, 1], repeat=n_qubits))

            tebd = ChainTEBD.from_optimizer(opt)
            return tebd.witness_open(ensemble, pulse, dt_ns=1.0, chi_max=chi_max, n_traj=n_traj)

        worker = Worker(mps_task)
        worker.signals.result.connect(self.on_mps_success)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(lambda: self._reset_btn(self.run_mps_btn, "Run MPS Witness"))
        self.threadpool.start(worker)

    def on_mps_success(self, mps_res):
        text = "--- MPS Witness Results ---\n"
        for k, v in mps_res.items():
            text += f"{k}: {v}\n"
        self.mps_output.setPlainText(text)
