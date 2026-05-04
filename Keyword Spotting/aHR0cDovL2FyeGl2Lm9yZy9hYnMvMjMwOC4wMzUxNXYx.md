# KEYWORD SPOTTING SIMPLIFIED: A SEGMENTATION-FREE APPROACH USING CHARACTER COUNTING AND CTC RE-SCORING

George Retsinas, Giorgos Sfikas, Christophoros Nikou (2023)

## 🧩 Problem to Solve

본 논문은 문서 이미지 내에서 특정 키워드를 찾는 **Segmentation-free Keyword Spotting (KWS)** 문제를 해결하고자 한다. 일반적인 KWS는 문서를 단어 혹은 라인 단위로 먼저 분할(Segmentation)한 뒤 검색하는 방식을 취하지만, 실제 문서 이미지에서는 표(table)나 여백의 메모(marginalia)와 같은 복잡한 구조로 인해 정확한 사전 분할이 매우 어렵다.

최근의 Segmentation-free 방식들은 최신 객체 탐지(Object Detection) 시스템을 도입하여 단어의 Bounding Box를 제안하고 동시에 표현형(Representation)을 계산하는 방식을 사용한다. 그러나 이러한 방법들은 모델이 너무 크고 복잡하며, 대량의 합성 데이터(Synthetic data)를 통한 문서 수준의 학습이 필요하다는 단점이 있다. 따라서 본 연구의 목표는 복잡하고 무거운 DNN 모델 없이도, 효율적으로 문서 이미지를 스캔하여 쿼리 정보가 포함된 영역을 찾아내는 단순하고 가벼운 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **문자 발생 횟수 계산(Character Counting)**을 통해 Bounding Box를 효율적으로 추정하는 것이다. 주요 기여 사항은 다음과 같다.

1. **효율적인 Segmentation-free KWS 시스템 제안**: 픽셀 단위의 문자 존재 확률과 스케일 맵(Scale map)을 통해 문자 수를 계산하는 단순한 구조의 CNN 모델을 제안한다.
2. **효율적인 문서 스캔 알고리즘**: Integral Images(적분 이미지)와 Binary Search(이진 탐색)를 결합하여, 계산 비용을 최소화하면서 쿼리 문자의 수와 일치하는 영역을 빠르게 탐색한다.
3. **단계적 재점수화(Re-scoring) 파이프라인**: 단순 카운팅의 한계(문자 순서 무시 등)를 극복하기 위해 Pyramidal Representation과 CTC(Connectionist Temporal Classification) 기반의 Re-scoring 알고리즘을 도입하여 검색 정확도를 높인다.
4. **새로운 중첩 지표 $\text{IoW}$ 제안**: 기존의 $\text{IoU}$ (Intersection over Union)가 KWS 작업에서 너무 엄격하여 정답을 포함하고 있음에도 낮게 평가되는 문제를 해결하기 위해, 인접 단어를 침범하지 않는 한 박스 확장을 허용하는 $\text{IoW}$ (Intersection over Word) 지표를 제안한다.

## 📎 Related Works

KWS 시스템은 쿼리 방식에 따라 Query-by-Example (QbE)과 Query-by-String (QbS)으로 나뉘며, 본 논문은 QbS에 집중한다.

- **Segmentation-based Methods**: PHOC (Pyramidal Histogram of Characters)와 같은 속성 기반 표현형을 사용하여 이미 분할된 단어 이미지에서 검색을 수행한다. 하지만 사전 분할 단계의 오류가 전체 성능에 영향을 미친다.
- **Segmentation-free Methods**: 
    - **Sliding Window**: 단순하지만 계산 효율성이 낮다.
    - **Region Proposal**: 객체 탐지 프레임워크를 사용하여 후보 영역을 먼저 생성한 뒤 랭킹을 매긴다. 성능은 좋으나 모델이 무겁고 학습 과정이 복잡하다.
- **차별점**: 본 논문은 객체 탐지 방식의 복잡한 Prediction Head를 제거하고, 단순한 '문자 카운팅'이라는 직관적인 접근법을 통해 모델의 경량화와 추론 속도를 동시에 달성했다.

## 🛠️ Methodology

### 1. Character Counting Network 학습
모델은 입력 이미지를 받아 문자 확률 맵 $F$와 스케일 맵 $S$를 생성한다.

- **아키텍처**: Lightweight ResNet-like backbone을 사용하며, 이후 두 개의 분기(CNN Decoder와 CNN Scaler)로 나뉜다.
    - **CNN Decoder**: 각 픽셀에서 어떤 문자가 존재할 확률을 나타내는 3D 텐서 $F \in \mathbb{R}^{H_r \times W_r \times C}$를 생성한다.
    - **CNN Scaler**: 각 픽셀이 해당 문자의 어느 정도 비중을 차지하는지를 나타내는 스케일 맵 $S \in \mathbb{R}^{H_r \times W_r}$를 생성한다. (출력값은 Sigmoid를 통해 $0 \sim 1$ 사이로 제한된다.)
- **수식**: 스케일이 적용된 피처 맵 $F_s$는 다음과 같이 정의된다.
  $$F_s[i, j, k] = F[i, j, k] \cdot S[i, j]$$
  특정 Bounding Box $(s_i, s_j)$에서 $(e_i, e_j)$까지의 문자 발생 횟수 $y^c$는 다음과 같이 계산된다.
  $$y^c = \sum_{i=s_i}^{e_i} \sum_{j=s_j}^{e_j} F_s(i, j)$$
- **손실 함수**: CTC loss와 Counting regression loss를 함께 사용한다.
  $$\mathcal{L} = \mathcal{L}_{CTC} + 10 \cdot \mathcal{L}_{count}$$
  여기서 $\mathcal{L}_{count} = \|y^c - t^c\|^2$ 이며, $t^c$는 실제 정답 문자 히스토그램이다.

### 2. 효율적인 Bounding Box 추정 (Spotting)
단순한 Sliding Window는 $O(N_r^3)$의 복잡도를 가지므로, 본 논문은 이를 $O(N_r \log N_r)$ 수준으로 낮춘다.

1. **Integral Images**: 합산 연산을 $O(1)$으로 처리하기 위해 적분 이미지를 사용한다.
2. **Binary Search**: 
    - 먼저 스케일 맵 $S$에서 문자 1개 분량의 크기를 갖는 영역을 찾아 높이(Height)를 추정한다.
    - 추정된 높이를 바탕으로, 쿼리 문자 수와 일치하는 너비(Width)를 이진 탐색으로 빠르게 찾는다.
3. **Candidate Pruning**: 
    - 쿼리의 첫 번째 문자가 나타날 확률이 낮은 지점은 제외한다.
    - Bounding Box 내부의 문자 분포가 중앙에 집중되지 않은 경우(여러 라인에 걸쳐 있는 경우)를 제외하여 정확도를 높인다.

### 3. Similarity Scoring 및 Re-scoring
카운팅 기반 방식은 문자의 순서를 구분하지 못하는 문제(예: "and"와 "dan"을 동일하게 인식)가 있다. 이를 위해 두 단계의 정제 과정을 거친다.

- **1단계: Pyramidal Counting**: 쿼리 문자열을 여러 레벨의 구간으로 나누어 각 구간별 문자 수를 계산하고 코사인 유사도를 측정한다.
- **2단계: CTC Re-scoring**: 상위 $K$개의 후보 영역에 대해 CTC 기반의 Forced Alignment 방식을 적용한다. 단순한 마지막 단계의 스코어만 보는 것이 아니라, 쿼리 시퀀스가 모두 인식되는 최적의 지점을 찾아 스코어를 매기며, 이 과정에서 Bounding Box의 시작과 끝 지점을 미세하게 조정하여 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: IAM (필기체 영어) 및 George Washington (GW) 문서 데이터셋.
- **평가 지표**: Mean Average Precision (mAP).
- **중첩 지표**: $\text{IoU}$, $\text{x-IoU}$ (x축 중심), $\text{IoW}$ (제안된 지표)를 사용하였다.

### 주요 결과
- **SOTA 비교**: 제안된 방법은 특히 데이터가 적은 환경(GW 5-15)과 IAM 데이터셋에서 $\text{IoU} \ge 25\%$ 기준일 때, 매우 복잡한 모델인 Ctrl-F-Net보다 우수한 성능을 보였다.
- **지표 분석**: $\text{IoU}$ 임계값이 높아질수록 성능이 급격히 하락하는 현상이 발견되었으나, 제안한 $\text{IoW}$ 지표를 적용했을 때는 높은 임계값에서도 성능이 견고하게 유지되었다. 이는 모델의 탐지 능력 문제가 아니라 $\text{IoU}$ 지표의 엄격함 때문임을 시사한다.
- **Ablation Study**: 
    - 단순 카운팅보다는 CTC Re-scoring을 추가했을 때 mAP가 비약적으로 상승하였다.
    - Two-way CTC (양방향 조정) 방식이 가장 높은 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 딥러닝 모델의 복잡성을 높이는 대신, **Integral Images와 Binary Search**라는 고전적인 컴퓨터 비전 기법과 **CTC Re-scoring**이라는 효율적인 후처리 단계를 결합함으로써 성능과 속도를 모두 잡았다. 특히 $\sim 6\text{M}$ 파라미터라는 매우 가벼운 모델로 SOTA 수준의 성능을 낸 점이 고무적이다.

### 한계 및 비판적 해석
1. **Bounding Box의 정밀도**: 제안된 방식은 '문자 수'에 의존하므로 박스가 실제 단어보다 크게 예측되는 경향(Over-estimation)이 있다. CTC Re-scoring으로 이를 어느 정도 완화했지만, 근본적으로 Tight한 박스를 생성하는 메커니즘은 부족하다.
2. **공백 문자 처리 부재**: 공백(Space) 문자를 학습하지 않았기 때문에, 단어의 일부만 찾는 Sub-word 탐지가 빈번하게 발생한다. 이는 사용자 요구에 따라 장점이 될 수도 있으나, 엄격한 단어 단위 검색에서는 오류로 작용한다.
3. **GPU 최적화 미비**: 현재 추론 단계의 Binary Search 등이 CPU(Numba)에서 구현되어 있어, 전체 파이프라인의 GPU 가속화가 이루어진다면 더 큰 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 복잡한 객체 탐지 모델 대신 **문자 발생 횟수를 계산하는 단순한 CNN 모델**과 **적분 이미지 기반의 효율적인 스캔 알고리즘**을 사용하여 Segmentation-free Keyword Spotting을 구현하였다. 특히 **CTC 기반의 재점수화**를 통해 문자 순서 모호성을 해결하고, KWS 작업에 더 적합한 **$\text{IoW}$라는 새로운 중첩 지표**를 제안하였다. 이 연구는 모델의 경량화와 효율적인 알고리즘 설계만으로도 충분히 높은 성능의 문서 검색 시스템을 구축할 수 있음을 입증하였으며, 향후 실시간 문서 분석 시스템에 적용될 가능성이 높다.