# Learning specialized activation functions with the Piecewise Linear Unit

Yucong Zhou, Zezhou Zhu, Zhao Zhong (2021)

## 🧩 Problem to Solve

딥러닝 모델에서 활성화 함수(activation function)의 선택은 모델의 표현력과 최적화 성능에 결정적인 영향을 미친다. 기존의 ReLU 및 그 변형들은 널리 사용되고 있으며, 자동 탐색(automated search)을 통해 발견된 Swish와 같은 함수들은 특정 데이터셋에서 ReLU보다 우수한 성능을 보인다. 하지만 이러한 기존 방식들은 다음과 같은 한계를 가진다.

첫째, Swish와 같은 자동 탐색 기반 활성화 함수는 트리 구조의 탐색 공간(tree-based search space)을 사용하는데, 이는 매우 이산적(discrete)이고 제한적이어서 최적의 함수를 찾기가 어렵다. 둘째, 샘플 기반의 탐색 방법은 수백에서 수천 개의 후보 함수를 평가해야 하므로 계산 비용이 매우 높다. 이로 인해 각 데이터셋이나 신경망 구조에 최적화된 '전문화된(specialized)' 활성화 함수를 개별적으로 찾는 것이 현실적으로 불가능하며, 결국 검색된 하나의 함수(Swish)를 모든 상황에 범용적으로 사용하는 수준에 그치고 있다.

따라서 본 논문의 목표는 각 데이터셋과 모델 아키텍처에 맞춰 유연하게 학습될 수 있으면서도, 추론 효율성이 높고 학습이 안정적인 새로운 활성화 함수인 Piecewise Linear Unit(PWLU)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 활성화 함수를 **분할 선형 함수(piecewise linear function)** 형태로 설계하여, 이를 통한 범용 근사(universal approximation) 능력을 확보하고 경사 하강법(gradient descent)으로 직접 최적화하는 것이다.

특히, 단순히 파라미터화된 함수를 사용하는 것을 넘어, 학습 과정에서 활성화 함수의 유연한 영역이 실제 입력 데이터의 분포와 일치하지 않아 학습 효율이 떨어지는 **입력 경계 불일치(input-boundary misalignment)** 문제를 식별하였다. 이를 해결하기 위해 입력 데이터의 통계량을 활용하여 경계를 재조정하는 **통계 기반 재정렬(statistic-based realignment)** 방법론을 제안하여 학습의 안정성과 성능을 극대화하였다.

## 📎 Related Works

본 논문에서는 기존 활성화 함수 연구를 세 가지 범주로 분류하여 설명한다.

1.  **고정 형태 활성화 함수 (Fixed-shape activation functions):** ReLU, Leaky ReLU, PReLU, ELU, SELU 등이 포함된다. 이들은 설계가 간단하지만 형태가 고정되어 있거나 학습 가능한 파라미터가 매우 적어, 다양한 데이터셋과 아키텍처에서 일관된 성능 향상을 보여주지 못하는 유연성의 한계가 있다.
2.  **유연한 활성화 함수 (Flexible activation functions):** Swish는 강화학습을 통해 탐색되었으나 앞서 언급한 대로 탐색 비용이 너무 크다. APL(Adaptive Piecewise Linear)과 PAU(Padé Activation Units)는 경사 하강법으로 최적화 가능한 유연한 수식을 사용하지만, 수식이 복잡하여 학습이 불안정하거나 추론 효율이 낮으며, 입력 분포와 파라미터 간의 정렬(alignment) 문제를 고려하지 않는다.
3.  **문맥 기반 활성화 함수 (Contextual-based activation functions):** Dynamic-ReLU나 Funnel-ReLU와 같이 입력의 전역적 또는 지역적 문맥을 활용하는 many-to-one 함수들이 제안되었다. 이들은 모델 용량을 증가시키지만, PWLU는 이와 대조적으로 단순한 스칼라(scalar) 함수 형태에 집중하여 범용성을 높였다.

## 🛠️ Methodology

### 1. Piecewise Linear Unit (PWLU) 정의
PWLU는 입력 공간을 여러 개의 구간으로 나누어 각 구간에서 선형 함수를 적용하는 구조이다. 주요 파라미터는 다음과 같다.
- 구간의 수 $N$ (하이퍼파라미터)
- 왼쪽 경계 $B_L$ 및 오른쪽 경계 $B_R$
- $N+1$개의 구분점에서의 $y$축 값 $Y^P$
- 최좌측 기울기 $K_L$ 및 최우측 기울기 $K_R$

전방 계산(forward pass) 수식은 다음과 같다.

$$
\text{PWLU}_N(x, B_L, B_R, Y^P, K_L, K_R) = 
\begin{cases} 
(x - B_L) \cdot K_L + Y_0^P & x < B_L \\ 
(x - B_R) \cdot K_R + Y_N^P & x \ge B_R \\ 
(x - B_{idx}) \cdot K_{idx} + Y_{idx}^P & B_L \le x < B_R 
\end{cases}
$$

여기서 $idx$는 $x$가 속한 구간의 인덱스이며, 구간 길이 $d = (B_R - B_L)/N$을 이용하여 다음과 같이 계산된다.
- $idx = \lfloor (x - B_L) / d \rfloor$
- $B_{idx} = B_L + idx \cdot d$
- $K_{idx} = (Y_{idx+1}^P - Y_{idx}^P) / d$

### 2. 학습 방법: 통계 기반 재정렬 (Statistic-based Realignment)
PWLU의 유연성은 주로 $[B_L, B_R]$ 영역에 집중되어 있다. 만약 이 영역이 실제 입력 분포와 어긋나 있다면, 대부분의 입력값이 $K_L$ 또는 $K_R$ 영역에 속하게 되어 학습 가능한 파라미터 $Y^P$가 제대로 활용되지 못한다. 이를 해결하기 위해 학습을 두 단계로 나눈다.

- **Phase I (통계 수집 단계):** 
    - 모든 PWLU를 ReLU 형태로 초기화한다.
    - 일반적인 학습을 진행하되, PWLU의 파라미터는 업데이트하지 않고 고정한다.
    - 대신, 각 PWLU로 들어오는 입력 $x$의 이동 평균(running mean) $\mu$와 이동 표준편차(running std) $\sigma$를 수집한다.
    - $\mu = \mu \cdot 0.9 + \text{mean}(x) \cdot 0.1$
    - $\sigma = \sigma \cdot 0.9 + \text{std}(x) \cdot 0.1$

- **Phase II (본 학습 단계):** 
    - 수집된 통계량을 바탕으로 경계를 다음과 같이 재설정한다.
    - $B_L = \mu - 3\sigma, \quad B_R = \mu + 3\sigma$
    - 이후 $K_L=0, K_R=1$ 및 $Y_{idx}^P = \text{ReLU}(B_{idx})$로 설정하여 다시 ReLU 형태로 시작한다.
    - 이제 경계가 입력 분포에 정렬되었으므로, 경사 하강법을 통해 PWLU의 모든 파라미터를 학습시킨다.

### 3. 추론 효율화
추론 시에는 매번 구간을 계산하는 대신, 미리 계산 가능한 값들을 상수로 치환하여 연산량을 줄인다.
수식을 $x \cdot S_{idx} + O_{idx}$ 형태로 재작성하며, 여기서 $S_{idx} = K_{idx}$이고 $O_{idx} = Y_{idx}^P - B_{idx} \cdot K_{idx}$이다. 이를 통해 단순한 곱셈-덧셈(Multiply-Add) 연산만으로 추론이 가능하게 하여 ReLU나 Swish에 근접한 속도를 구현하였다.

## 📊 Results

### 1. ImageNet 분류 실험
ResNet-18/50, MobileNet-V2/V3, EfficientNet-B0 등 5가지 아키텍처에서 실험을 진행하였다. 
- **결과:** PWLU는 모든 아키텍처에서 ReLU 및 Swish보다 우수한 Top-1 정확도를 달성하였다. 
- 특히 Swish 대비 향상 폭은 ResNet-18(0.9%), ResNet-50(0.53%), MobileNet-V2(1.0%), MobileNet-V3(1.7%), EfficientNet-B0(1.0%)로 나타났다.
- 다른 유연한 함수(APL, PAU)들은 아키텍처별로 성능 편차가 컸으나, PWLU는 일관되게 높은 성능을 보였다.

### 2. COCO 객체 검출 실험
Mask R-CNN과 RetinaNet 프레임워크를 사용하여 성능을 검증하였다.
- **결과:** Swish가 ReLU 대비 약 0.8%~1% AP 향상을 보인 반면, PWLU는 Swish보다 추가로 0.4%~0.5% AP를 더 향상시켰다. 이는 PWLU가 이미지 분류뿐만 아니라 검출 작업에서도 일반화 능력이 뛰어남을 입증한다.

### 3. 추론 속도 및 효율성
- **추론 시간:** NVIDIA V100 GPU 기준, PWLU의 추론 시간은 ReLU 및 Swish와 거의 동일하며, 복잡한 함수인 APL이나 PAU보다 훨씬 빠르다.
- **학습 시간:** 더 많은 경사도(gradient) 항을 계산해야 하므로 ReLU 대비 학습 속도는 약 20% 느리다.

## 🧠 Insights & Discussion

본 논문의 결과는 **활성화 함수가 모델의 구조와 데이터의 특성에 따라 전문화(specialization)될 때 더 높은 성능을 낼 수 있음**을 시사한다. 

- **강점:** PWLU는 단순한 수식 구조 덕분에 추론 효율성이 매우 높으면서도, 통계 기반 재정렬을 통해 학습의 불안정성을 제거하였다. 시각화 결과, 학습된 PWLU는 기존의 핸드메이드 함수에서는 보기 힘든 V-자 형태 등의 다양한 모양을 띠며, 층(layer)마다 서로 다른 형태를 학습함을 확인하였다.
- **한계 및 논의:** 구간의 수 $N$이 너무 많아지면($N=20$ 이상), 각 구간에 할당되는 데이터 포인트가 적어져 경사도의 분산이 커지고 오히려 성능이 소폭 하락하는 경향이 발견되었다. 이는 파라미터 수의 증가가 반드시 성능 향상으로 이어지지 않으며, 적절한 $N$의 범위($8 \sim 16$)를 설정하는 것이 중요함을 보여준다.
- **비판적 해석:** 본 연구는 스칼라 함수에 집중하였으나, 최근의 Dynamic-ReLU와 같은 문맥 기반 함수들의 성능 향상 폭이 매우 크다는 점을 고려할 때, PWLU의 구조에 이러한 동적 메커니즘을 결합한다면 더 큰 성능 향상을 기대할 수 있을 것이다.

## 📌 TL;DR

PWLU는 분할 선형 함수를 기반으로 각 모델과 데이터셋에 최적화된 활성화 함수를 학습하는 방법론이다. 입력 데이터의 분포를 분석해 학습 영역을 정렬하는 '통계 기반 재정렬' 기법을 통해 학습 안정성을 확보하였으며, 추론 시에는 단순한 선형 연산으로 최적화하여 효율성을 극대화하였다. ImageNet과 COCO 데이터셋에서 Swish를 포함한 기존 함수들을 일관되게 능가함으로써, 전문화된 활성화 함수의 실용성과 효과를 입증하였다.