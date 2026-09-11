"""ROS2 node: publishes the Mac-side perception agent's JSON hand-off file on
a ROS2 topic inside the VM.

The Mac has no rclpy (ROS2 lives only in this Ubuntu VM, by this project's
own architecture decision). Grounding DINO inference and 3D pose estimation
run on the Mac (scripts/run_perception_agent.py in the main repo, MPS-
accelerated) and are handed off as a JSON file, scp'd into this VM. This
node's only job: watch that JSON file, and whenever its contents change,
republish it verbatim as a std_msgs/String on /perception/detections so
downstream ROS2 agents (Grounding, Execution) can subscribe.

NOT TESTED on the machine this was written on -- no ROS2 environment exists
there. First real test happens in this Ubuntu/Humble VM. Written to be
idiomatic Humble rclpy from API knowledge; verify with:
    ros2 run agentic_grasp_stack_perception detection_publisher
    ros2 topic echo /perception/detections
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

DEFAULT_JSON_PATH = str(Path.home() / "Agentic-grasp-stack" / "outputs" / "latest_detections.json")
DEFAULT_TOPIC = "/perception/detections"
DEFAULT_POLL_PERIOD_SEC = 2.0


class DetectionPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("detection_publisher")

        self.declare_parameter("json_path", DEFAULT_JSON_PATH)
        self.declare_parameter("topic", DEFAULT_TOPIC)
        self.declare_parameter("poll_period_sec", DEFAULT_POLL_PERIOD_SEC)

        self._json_path = Path(
            self.get_parameter("json_path").get_parameter_value().string_value
        )
        topic = self.get_parameter("topic").get_parameter_value().string_value
        poll_period = self.get_parameter("poll_period_sec").get_parameter_value().double_value

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(String, topic, qos)

        self._last_mtime: float | None = None
        self._warned_missing = False

        self.create_timer(poll_period, self._poll_and_publish)

        self.get_logger().info(
            f"watching {self._json_path} every {poll_period}s, publishing to {topic}"
        )

    def _poll_and_publish(self) -> None:
        if not self._json_path.exists():
            if not self._warned_missing:
                self.get_logger().warn(f"{self._json_path} does not exist yet, waiting")
                self._warned_missing = True
            return
        self._warned_missing = False

        mtime = self._json_path.stat().st_mtime
        if mtime == self._last_mtime:
            return  # unchanged since last publish -- don't spam identical data

        raw_text = self._json_path.read_text()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"{self._json_path} is not valid JSON, skipping: {exc}")
            return

        self._last_mtime = mtime
        self._publisher.publish(String(data=raw_text))
        num_detections = len(payload.get("detections", []))
        self.get_logger().info(f"published {num_detections} detections (file changed)")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DetectionPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
