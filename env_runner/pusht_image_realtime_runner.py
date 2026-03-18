#!/usr/bin/env python3
from gym_pusht.envs.pusht_image_env import PushTImageEnv
import numpy as np
import time
import cv2
import os
import threading
from datetime import datetime
import tqdm

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PushTImageRealtimeRunner(Node):
    def __init__(
        self,
        render_size: int = 96,
        fps: int = 10,
        action_topic: str = "pusht/action",
        state_topic: str = "pusht/state",
        image_topic: str = "pusht/image",
        init_topic: str = "pusht/init",
        enable_human_render: bool = True,
        perturb_level: float = 1.5,
        max_steps: int = 300,
        max_episodes: int = 10,
    ):
        super().__init__("pusht_image_realtime_runner")

        self.render_size = render_size
        self.fps = float(fps)
        self.period = 1.0 / self.fps

        self.action_topic = action_topic
        self.state_topic = state_topic
        self.image_topic = image_topic
        self.init_topic = init_topic

        self.enable_human_render = enable_human_render
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.perturb_level = perturb_level
        self.episode_count = 0

        # Env
        self.seed = 100000
        print(f"Initializing environment with seed {self.seed}")
        PushTImageEnv.metadata["video.frames_per_second"] = int(fps)
        self.env = PushTImageEnv(render_size=render_size, perturb_level=self.perturb_level)
        self.env.seed(self.seed)
        self.obs = self.env.reset()

        if self.enable_human_render:
            self.env._render_frame("human")

        self.bridge = CvBridge()

        self._act_lock = threading.Lock()
        self.latest_ros_action = None

        self.recording = False
        self.video_writer = None
        self.record_dir = "recordings"
        os.makedirs(self.record_dir, exist_ok=True)

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.sub_action = self.create_subscription(Float32MultiArray, self.action_topic, self._command_callback, qos_profile=qos)
        self.state_pub = self.create_publisher(Float32MultiArray, self.state_topic, qos_profile=qos)
        self.image_pub = self.create_publisher(Image, self.image_topic, qos_profile=qos)
        self.init_pub = self.create_publisher(Bool, self.init_topic, qos_profile=qos)

        self.get_logger().info(
            f"Started PushTImageRealtimeRunner fixed-rate loop (fps={fps}) "
            f"action={action_topic}, state={state_topic}, image={image_topic}, init={init_topic}"
        )

    # ---------------- ROS callbacks ----------------
    def _command_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        try:
            action_array = np.array(msg.data[:2], dtype=np.float32)
            if np.any(np.isnan(action_array)):
                return
            with self._act_lock:
                self.latest_ros_action = np.clip(action_array, 0, self.env.window_size)
        except Exception:
            pass

    def _select_action(self):
        with self._act_lock:
            return self.latest_ros_action.copy() if self.latest_ros_action is not None else self.obs["agent_pos"]

    # ---------------- publish helpers ----------------
    def _publish_state(self, obs):
        msg = Float32MultiArray()
        msg.data = [float(obs["agent_pos"][0]), float(obs["agent_pos"][1])]
        self.state_pub.publish(msg)

    def _publish_image(self, obs):
        img_bgr = np.transpose((obs["image"] * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))[..., ::-1]
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(msg)

    def _publish_init(self):
        self.init_pub.publish(Bool(data=True))

    def start_recording(self, filename=None):
        if self.recording:
            return
        filename = filename or f"recording_{self.seed}_{self.perturb_level}.mp4"
        h, w = self.env._render_frame("rgb_array").shape[:2]
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.record_dir, filename), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        self.recording = True

    def stop_recording(self):
        if not self.recording:
            return
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        self.recording = False

    def _record_frame_if_needed(self):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(self.env._render_frame("rgb_array")[..., ::-1])

    def loop(self):
        next_t = time.perf_counter()
        step = 0

        self._publish_init()
        self.start_recording()

        all_max_rewards = []
        max_reward = 0.0
        pbar = tqdm.tqdm(range(self.max_episodes), desc="Episodes")

        while rclpy.ok():
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            next_t += self.period

            action = self._select_action()
            self.obs, reward, done, info = self.env.step(action)
            step += 1
            max_reward = max(max_reward, reward)
            pbar.set_postfix({"reward": f"{reward:.4f}", "max": f"{max_reward:.4f}", "step": f"{step}/{self.max_steps}"})
            self._publish_state(self.obs)
            self._publish_image(self.obs)
            
            if self.enable_human_render:
                self.env._render_frame("human")
            self._record_frame_if_needed()

            if done or step >= self.max_steps:
                self._record_frame_if_needed()
                self.stop_recording()
                self.episode_count += 1
                all_max_rewards.append(max_reward)
                max_reward = step = 0
                pbar.update(1)
                if self.episode_count >= self.max_episodes:
                    break
                self.seed += 1
                self.env.seed(self.seed)
                self.start_recording()
                print(f"Resetting environment with seed {self.seed}")
                self.obs = self.env.reset()
                with self._act_lock:
                    self.latest_ros_action = None
                self._publish_init()
                if self.enable_human_render:
                    self.env._render_frame("human")

        if all_max_rewards:
            mean_reward = np.mean(all_max_rewards)
            std_reward = np.std(all_max_rewards)
            stats = "\n".join([
                "="*50,
                f"All {self.max_episodes} episodes completed!",
                f"Mean max_reward: {mean_reward:.4f}",
                f"Std max_reward: {std_reward:.4f}",
                f"Individual max_rewards: {[f'{r:.4f}' for r in all_max_rewards]}",
                f"Policy: ddim",
                # f"Policy: flow matching",
                f"perturb: {self.perturb_level}",
                f"mode: realtime_naive_async",
                "="*50
            ])
            print(f"\n{stats}")
            with open(os.path.join(self.record_dir, "result.txt"), 'w') as f:
                f.write(stats + "\n")

    def close(self):
        self.stop_recording()
        try:
            self.env.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PushTImageRealtimeRunner()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.loop()
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
