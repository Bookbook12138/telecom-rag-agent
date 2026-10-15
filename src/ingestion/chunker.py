import re
from dataclasses import dataclass
from typing import List

# 匹配 "3.2 基于机器学习的协作频谱感知技术" 或 "## 3.2 xxx" 或 "3.2.1 xxx"
SECTION_PATTERN = re.compile(
    r'^(?:#{1,4}\s*)?(\d{1,2}(?:\.\d{1,3}){0,3})\s+([^\n]{2,80})$',
    re.MULTILINE
)

@dataclass
class Chunk:
    text: str
    section_id: str
    section_title: str
    source: str
    chunk_index: int


def split_by_sections(full_text: str, source: str,
                      max_chunk_size: int = 1200,
                      overlap: int = 100) -> List[Chunk]:
    matches = list(SECTION_PATTERN.finditer(full_text))
    matches = [m for m in matches if _is_valid_section(m.group(1))]
    if not matches:
        return _fallback_split(full_text, source, max_chunk_size, overlap)

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_text = full_text[start:end].strip()
        section_id = m.group(1)
        section_title = m.group(2).strip()

        if len(section_text) <= max_chunk_size:
            chunks.append(Chunk(
                text=section_text,
                section_id=section_id,
                section_title=section_title,
                source=source,
                chunk_index=len(chunks)
            ))
        else:
            for sub in _sliding_window(section_text, max_chunk_size, overlap):
                chunks.append(Chunk(
                    text=sub,
                    section_id=section_id,
                    section_title=section_title,
                    source=source,
                    chunk_index=len(chunks)
                ))
    return chunks


def _sliding_window(text, size, overlap):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def _fallback_split(text, source, size, overlap):
    return [
        Chunk(text=t, section_id="", section_title="",
              source=source, chunk_index=i)
        for i, t in enumerate(_sliding_window(text, size, overlap))
    ]


# 保留旧接口，避免其他地方 import 报错
def split_text(text: str, chunk_size: int = 800, overlap: int = 100):
    return _sliding_window(text, chunk_size, overlap)


def _is_valid_section(section_id: str) -> bool:
    # 排除 4 位数字（年份，如 2030）
    if len(section_id.replace(".", "")) >= 4 and "." not in section_id:
        return False
    # 排除太大的数字
    first = int(section_id.split(".")[0])
    if first > 20:
        return False
    return True