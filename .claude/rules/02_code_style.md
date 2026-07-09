# Rule 02 — 코드 스타일

## Python 기본 규칙
- PEP 8 준수, 라인 길이 최대 100자
- 변수명은 snake_case, 클래스명은 PascalCase
- 매직 넘버 사용 금지 — 상수로 분리

## pandas 작성 규칙
- 메서드 체이닝은 괄호로 감싸서 줄바꿈 허용
  ```python
  result = (
      df
      .query("age > 20")
      .groupby("category")
      .agg({"value": "sum"})
      .reset_index()
  )
  ```
- `.apply(lambda ...)` 보다 벡터 연산 우선 사용
- `iterrows()` 사용 지양 — 대신 `vectorized` 또는 `itertuples()` 사용

## 임포트 순서
```python
# 1. 표준 라이브러리
import os
from pathlib import Path

# 2. 서드파티
import numpy as np
import pandas as pd

# 3. 시각화
import matplotlib.pyplot as plt
import seaborn as sns
```

## 함수 작성
- 분석 단계별로 함수로 분리하여 재사용 가능하게 작성
- 함수 첫 줄에 한 줄 docstring 필수
- 반환값 타입 힌트 권장

## 주석
- 코드 블록 단위로 한국어 주석 허용
- `# TODO:`, `# FIXME:` 태그 사용
