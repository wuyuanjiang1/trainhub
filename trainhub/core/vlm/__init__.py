"""视觉大模型（VLM）预标注后端。

调用 OpenAI 兼容接口生成候选标注：所有服务商预设共用同一套请求与解析
代码（DeepSeek / 智谱 / 百炼 / 硅基流动 / 自定义），模型以 0-1000 归一化
坐标输出矩形框，换算回像素后产出与 labelme 兼容的 shape 字典。

模块划分：

- :mod:`.providers` —— 服务商预设、API Key 的环境变量查找、VLMError
- :mod:`.prompt` —— 检测提示词构造与"裸目标名清单"白名单解析
- :mod:`.parse` —— 模型回复解析、坐标换算、标注词归一化
- :mod:`.client` —— OpenAI 兼容请求、测试连接、总入口 ``predict_shapes_vlm``
- :mod:`.tracker` —— token 用量 / 检出率 / 费用统计

注意：VLM 输出的是矩形框；segment 任务的多边形候选需在画布上人工调整，
后续可接"VLM 出框 + SAM 出掩膜"的两阶段管线。全部请求走标准库 urllib，
不引入新依赖。
"""

from .client import predict_shapes_vlm
from .client import test_connection
from .parse import normalize_english_label
from .parse import parse_detections
from .prompt import DEFAULT_DETECT_PROMPT
from .prompt import build_prompt
from .prompt import extract_label_whitelist
from .providers import ENV_KEYS
from .providers import VLMError
from .providers import VLMProvider
from .providers import VLM_PROVIDERS
from .providers import provider_env_key
from .providers import provider_from_key
from .tracker import UsageTracker

__all__ = [
    "DEFAULT_DETECT_PROMPT",
    "ENV_KEYS",
    "UsageTracker",
    "VLMError",
    "VLMProvider",
    "VLM_PROVIDERS",
    "build_prompt",
    "extract_label_whitelist",
    "normalize_english_label",
    "parse_detections",
    "predict_shapes_vlm",
    "provider_env_key",
    "provider_from_key",
    "test_connection",
]
