# SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking

Dongyan Guo, Jun Wang, Ying Cui, Zhenhua Wang, Shengyong Chen (2019)

## 🧩 Problem to Solve

본 논문은 비주얼 객체 추적(Visual Object Tracking) 분야에서 기존의 Siamese RPN(Region Proposal Network) 기반 추적기들이 가진 한계점을 해결하고자 한다. 기존의 SiamRPN, SiamRPN++, SPM과 같은 모델들은 객체의 위치와 크기를 예측하기 위해 Anchor box를 사용한다. 그러나 Anchor 기반 방식은 다음과 같은 문제점을 가진다:

1. **하이퍼파라미터 의존성**: Anchor의 개수, 크기, 종횡비(aspect ratio)를 설정하는 과정이 매우 까다로우며, 최적의 성능을 내기 위해 전문가의 세심한 튜닝이 필수적이다.
2. **유연성 부족**: 고정된 크기의 Anchor box를 사용하기 때문에, 객체의 급격한 형태 변형(deformation)이나 포즈 변화(pose variation)가 일어날 경우 이를 유연하게 대응하기 어렵다.

따라서 본 연구의 목표는 Anchor와 Region Proposal이 필요 없는(Anchor-free and Proposal-free) 단순하고 효율적인 Siamese 네트워크를 설계하여, 복잡한 튜닝 없이도 높은 정확도와 실시간 속도를 동시에 달성하는 것이다.

## ✨ Key Contributions

SiamCAR의 핵심 아이디어는 비주얼 추적 작업을 **픽셀 단위(per-pixel manner)**의 두 가지 하위 문제, 즉 **분류(Classification)**와 **회귀(Regression)**로 분해하여 해결하는 것이다.

- **픽셀 단위의 접근 방식**: 각 픽셀이 객체에 속하는지(분류)와 해당 픽셀에서 객체 경계까지의 거리(회귀)를 동시에 예측한다.
- **Anchor-free 구조**: 기존의 복잡한 Anchor 기반 설계를 완전히 제거함으로써 하이퍼파라미터 튜닝의 어려움을 없애고 모델 구조를 단순화하였다.
- **단일 응답 맵(Single Response Map) 활용**: RPN 모델들이 탐지와 회귀를 위해 별도의 맵을 사용하는 것과 달리, 하나의 고유한 응답 맵에서 위치와 바운딩 박스 정보를 직접 예측하여 효율성을 높였다.

## 📎 Related Works

최근의 비주얼 추적 연구는 주로 Siamese 네트워크 구조를 중심으로 발전해 왔다.

- **SiamFC 및 초기 모델**: 타겟 템플릿과 검색 영역 간의 유사도 맵을 학습하여 매칭 문제로 접근하였다. 하지만 객체의 크기 변화에 대응하기 위해 여러 스케일의 검색 영역을 처리해야 하므로 연산 비용이 높다는 단점이 있었다.
- **SiamRPN 및 발전 모델**: RPN을 도입하여 다중 스케일 특징 맵 추출 과정을 생략하고 효율성을 높였다. DaSiamRPN은 하드 네거티브(hard negative) 데이터를 추가하여 변별력을 높였고, SiamRPN++는 ResNet-50 백본을 도입하여 깊은 네트워크를 통해 성능을 끌어올렸다.
- **한계점**: 앞서 언급한 RPN 기반 모델들은 모두 Anchor에 의존한다. 반면, ECO와 같은 Anchor-free 추적기들이 존재하지만, 벤치마크 데이터셋(예: GOT-10K)에서 Anchor 기반 모델들에 비해 정확도와 속도 면에서 여전히 격차가 존재했다.

SiamCAR는 이러한 RPN 기반 모델의 성능적 이점과 Anchor-free 모델의 단순함을 결합하여, 튜닝 없이도 최신 SOTA(State-of-the-art) 성능을 내는 것을 목표로 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

SiamCAR는 크게 두 개의 서브 네트워크로 구성된다:

- **Siamese Subnetwork**: 특징 추출(Feature Extraction)을 담당한다.
- **Classification-Regression Subnetwork**: 특징 맵을 입력받아 바운딩 박스를 예측한다.

### 2. Siamese Subnetwork (특징 추출)

백본 네트워크로 수정된 ResNet-50을 사용하며, 패딩이 없는 Fully Convolutional 구조이다.

- **특징 융합**: 인식 능력과 변별력을 모두 확보하기 위해 ResNet-50의 마지막 세 개 잔차 블록(residual blocks)에서 추출된 특징 $F_3(X), F_4(X), F_5(X)$를 결합(Concatenate)한다.
  $$\phi(X) = \text{Cat}(F_3(X), F_4(X), F_5(X)) \quad (2)$$
- **Depth-wise Cross Correlation**: 템플릿 $\phi(Z)$와 검색 영역 $\phi(X)$ 사이에서 채널별 상관관계 연산을 수행하여 다채널 응답 맵 $R$을 생성한다.
  $$R = \phi(X) \circ \phi(Z) \quad (1)$$
- **차원 축소**: 이후 $1 \times 1$ 컨볼루션 커널을 통해 응답 맵의 차원을 256채널로 축소하여 연산 속도를 높인 $R^*$를 생성한다.

### 3. Classification and Regression Subnetwork (예측)

응답 맵 $R^*$의 각 위치 $(i, j)$는 입력 검색 영역의 좌표 $(x, y)$에 대응된다. 이 위치에서 세 가지 브랜치가 병렬로 작동한다.

- **Classification Branch**: 해당 픽셀이 전경(foreground)인지 배경(background)인지를 예측하여 $w \times h \times 2$ 크기의 맵 $A_{cls}$를 생성한다.
- **Regression Branch**: 해당 픽셀에서 바운딩 박스의 네 면까지의 거리 $\tilde{t}(i, j) = (\tilde{l}, \tilde{t}, \tilde{r}, \tilde{b})$를 예측한다.
  $$\tilde{l}=x-x_0, \quad \tilde{t}=y-y_0, \quad \tilde{r}=x_1-x, \quad \tilde{b}=y_1-y \quad (3)$$
  여기서 $(x_0, y_0)$는 좌상단, $(x_1, y_1)$은 우하단 좌표이다.
- **Center-ness Branch**: 객체의 중심에서 멀리 떨어진 픽셀의 예측값은 품질이 낮으므로, 이를 제거하기 위해 Center-ness 점수 $C(i, j)$를 계산한다.
  $$C(i, j) = I(\tilde{t}(i, j)) * \frac{\min(\tilde{l}, \tilde{r})}{\max(\tilde{l}, \tilde{r})} \times \frac{\min(\tilde{t}, \tilde{b})}{\max(\tilde{t}, \tilde{b})} \quad (6)$$

### 4. 학습 목표 및 손실 함수

전체 손실 함수 $L$은 다음과 같이 세 가지 손실의 가중 합으로 정의된다.
$$L = L_{cls} + \lambda_1 L_{cen} + \lambda_2 L_{reg} \quad (8)$$

- $L_{cls}$: 분류를 위한 Cross-Entropy Loss.
- $L_{cen}$: Center-ness 예측을 위한 Binary Cross-Entropy Loss.
- $L_{reg}$: 바운딩 박스 회귀를 위한 IOU Loss (Eq 4).
- 가중치 설정: $\lambda_1 = 1, \lambda_2 = 3$으로 설정하여 학습하였다.

### 5. 추론 및 추적 절차 (Tracking Phase)

1. **최적 위치 탐색**: 분류 점수($cls$)에 스케일 변화 페널티($p_{ij}$)와 코사인 윈도우($H$)를 적용하여 가장 높은 점수를 가진 픽셀 $q$를 찾는다.
   $$q = \arg \max_{i, j} \{(1 - \lambda_d) cls_{ij} \times p_{ij} + \lambda_d H\} \quad (9)$$
2. **결과 안정화**: 단일 픽셀 $q$만 사용하면 프레임 간 지터링(jittering)이 발생할 수 있다. 이를 방지하기 위해 $q$ 주변 $n=8$개 이웃 중 상위 $k=3$개 지점을 선택하여, 이들의 회귀 바운딩 박스를 가중 평균(weighted average)하여 최종 결과를 출력한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: GOT-10K, LaSOT, OTB-50, UAV123.
- **비교 대상**: SiamRPN++, SPM, ECO, SiamFC, MDNet 등 최신 추적기.
- **지표**: Average Overlap (AO), Success Rate (SR), Precision.

### 2. 주요 결과

- **GOT-10K**: 모든 지표에서 1위를 기록하였다. 특히 SiamRPN++ 대비 AO는 5.2%, $\text{SR}_{0.5}$는 5.4%, $\text{SR}_{0.75}$는 9.0% 향상되었다. 속도 또한 52.27 FPS로 실시간 성능을 확보하였다.
- **LaSOT**: 대규모 데이터셋에서도 SiamRPN++ 및 다른 베이스라인 모델들을 유의미하게 앞서며 뛰어난 일반화 성능을 입증하였다.
- **OTB-50**: 저해상도(low resolution), 배경 clutter, 평면 외 회전(out-of-plane rotation), 변형(deformation) 등의 어려운 조건에서 특히 강한 모습을 보였다.
- **UAV123**: 빠른 움직임과 큰 스케일 변화가 있는 환경에서도 SOTA 성능을 달성하였다.

## 🧠 Insights & Discussion

**강점 및 해석**

- SiamCAR는 Anchor-free 방식임에도 불구하고, 기존 Anchor 기반의 고성능 모델들보다 더 높은 정확도를 달성하였다. 이는 픽셀 단위의 분류-회귀 구조가 객체의 기하학적 변화를 더 유연하게 캡처할 수 있음을 시사한다.
- 특히 Center-ness 브랜치를 통해 Outlier를 효과적으로 제거함으로써, 바운딩 박스 회귀의 정밀도를 높인 점이 주효했다.
- 복잡한 하이퍼파라미터 튜닝 없이 단순한 구조만으로 성능을 낸 것은 실용적인 관점에서 매우 큰 이점이다.

**한계 및 논의사항**

- 본 논문은 오프라인 추적(offline tracking) 전략을 사용하며, 첫 프레임의 템플릿만을 고정해서 사용한다. 따라서 추적 도중 객체의 외형이 급격하게 변하는 경우에 대한 템플릿 업데이트 전략은 명시적으로 다루어지지 않았다.
- $\lambda_1, \lambda_2$와 같은 손실 함수 가중치나 $k, n$과 같은 추론 시 하이퍼파라미터가 실험적으로 설정되었는데, 이에 대한 이론적 근거보다는 경험적 결과에 의존한 부분이 있다.

## 📌 TL;DR

SiamCAR는 비주얼 추적 문제를 픽셀 단위의 **분류(Classification)와 회귀(Regression)로 분해**하여 해결하는 **Anchor-free Siamese 네트워크**이다. 복잡한 Anchor 설정 없이도 ResNet-50 백본과 Center-ness 브랜치를 활용해 **실시간 속도(52.27 FPS)와 SOTA 정확도**를 동시에 달성하였다. 이 연구는 복잡한 튜닝 과정 없이 단순한 구조만으로도 강력한 추적 성능을 낼 수 있음을 증명하였으며, 향후 Anchor-free 기반의 효율적인 추적 알고리즘 연구에 중요한 이정표가 될 것으로 보인다.
