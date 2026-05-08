# Deep Variational Instance Segmentation

Jialin Yuan, Chao Chen, Li Fuxin (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 이미지 내의 각 픽셀에 대해 클래스 라벨과 인스턴스 라벨을 동시에 부여하는 **Instance Segmentation**이다. 이 작업은 다음과 같은 세 가지 주요 어려움이 존재한다. 첫째, 동일한 범주에 속하는 서로 다른 인스턴스들은 외형이 매우 유사할 수 있다. 둘째, 예측 시점에 이미지 내 인스턴스의 개수를 미리 알 수 없다. 셋째, 인스턴스 라벨은 **Permutation-invariant**(순열 불변성) 특성을 가진다. 즉, 정답(Ground Truth, GT)에서 인스턴스 라벨의 숫자를 서로 바꾸더라도 실제 의미는 변하지 않으므로, 기존의 Cross-Entropy(CE) 손실 함수를 직접 적용할 수 없다.

기존의 SOTA 알고리즘들은 주로 Search-based 전략을 사용한다. 이는 이미지를 그리드로 나누어 제안 영역(Proposal)을 생성한 후, 이를 분류하고 경계를 정밀화하는 방식이다. 그러나 이러한 방식은 수천 개의 Proposal을 생성하고 검증해야 하므로 연산 비용이 매우 높고 속도가 느리다는 단점이 있다. 본 논문의 목표는 Proposal 생성 과정 없이 Fully Convolutional Network(FCN)를 통해 직접 인스턴스 라벨을 예측하는 효율적인 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 세그멘테이션 문제를 **Variational Relaxation**(변분 완화) 관점에서 접근하여, 조각별 상수(Piecewise-constant) 함수를 찾는 최적화 문제로 정의하는 것이다.

중심적인 설계 직관은 고전적인 **Mumford-Shah variational segmentation** 알고리즘을 딥러닝 프레임워크에 통합하는 것이다. 특히, 인스턴스 라벨의 순열 불변성 문제를 해결하기 위해 새로운 **Permutation-invariant loss**를 제안하여, FCN이 실수 값(Real-valued)의 인스턴스 맵을 직접 출력하도록 학습시킨다. 이를 통해 Proposal 기반의 탐색 과정 없이 한 번의 global glance만으로 인스턴스를 분리해낼 수 있는 end-to-end 구조를 구축하였다.

## 📎 Related Works

기존의 인스턴스 세그멘테이션 접근 방식은 크게 두 가지로 나뉜다.

1. **Search-based methods**: Mask R-CNN, FCIS와 같이 Anchor box를 기반으로 Proposal을 생성하고 이를 정밀화하는 방식이다. 이들은 정확도는 높으나 수많은 Proposal을 생성하고 Non-Maximum Suppression(NMS)과 같은 후처리를 거쳐야 하므로 실시간 성능 확보가 어렵다.
2. **Search-free methods**: Proposal 생성 없이 각 픽셀의 서로게이트(Surrogate) 라벨을 예측한 후 휴리스틱한 후처리를 통해 인스턴스를 분리하는 방식이다. 예를 들어, Metric learning 기반 방식($[23, 15]$)은 동일 인스턴스 픽셀 간의 거리를 가깝게 학습시킨다. 그러나 이러한 방식들은 배경(Background)과 전경(Foreground)의 구분이 모호하여 복잡한 후처리에 의존하는 경향이 있다.

본 연구는 이러한 기존 방식들과 달리, 변분법적 목적 함수를 통해 배경과 전경을 명확히 분리하는 Binary term을 도입하고, 정규화 항을 통해 경계선을 날카롭게 유지함으로써 후처리 의존도를 낮추고 속도를 획기적으로 높였다.

## 🛠️ Methodology

### 전체 시스템 구조

DVIS의 전체 파이프라인은 크게 세 단계로 구성된다: **FCN을 통한 실수 값 라벨 맵 예측** $\rightarrow$ **Mean-shift를 통한 이산화(Discretization)** $\rightarrow$ **분류 및 검증 네트워크를 통한 최종 필터링**.

### 훈련 목표 및 변분 목적 함수

본 논문은 인스턴스 세그멘테이션을 다음과 같은 에너지 최소화 문제로 정의한다.

$$F(f,C) = \int_{\Omega} L^b(f(x,y), \mathbb{I}_{[GT(x,y)=0]}) dxdy + \mu \int_{\Omega} \|\nabla f\|^2 dxdy + \nu |C| + \int_{\Omega} |f - \text{Round}(f)| dxdy + \int_{\Omega} \int_{\Omega} L^{pi}(|f(x_1, y_1) - f(x_2, y_2)|, \mathbb{I}_{[GT(x_1, y_1) \neq GT(x_2, y_2)]}) dx_1 dy_1 dx_2 dy_2$$

각 항의 역할은 다음과 같다.

1. **Binary Loss ($L^b$)**: 픽셀이 배경인지 전경인지를 구분한다. 배경 픽셀은 $0$ 이하로, 전경 픽셀은 일정 임계값 $m_1$ 이상으로 밀어내어 배경-전경을 명확히 분리한다.
2. **Regularization**: Mumford-Shah 모델에서 유래한 항으로, $\mu$ 항은 영역 내부의 매끄러움(Smoothness)을 유지하고, $\nu$ 항은 경계선 길이 $|C|$를 최소화하여 과잉 분할(Over-segmentation)을 방지한다. 실제 구현에서는 Cauchy loss 형태의 $L'_{MS}$로 근사하여 사용한다.
3. **Quantization**: 예측값 $f$가 정수 값에 가까워지도록 유도하여 인스턴스 간의 마진을 확보하고 후처리를 용이하게 한다.
4. **Permutation Invariant Loss ($L^{pi}$)**: 두 픽셀의 GT 라벨이 같으면 예측값의 차이를 $0$으로 만들고, 다르면 최소 $m_2$ 이상의 차이를 갖도록 강제한다. 이를 통해 구체적인 라벨 숫자와 관계없이 인스턴스 간의 구별 가능성만을 학습한다.

### 손실 함수의 세부 구현

모든 손실 함수는 이상치에 강건한 **Huber loss** $L^h(v, \theta)$를 기반으로 설계되었다.

- **Binary Loss**: $GT=0$이면 $\text{ReLU}(f)$를 최소화하고, $GT>0$이면 $\text{ReLU}(m_1 - f)$를 최소화한다.
- **Permutation Invariant Loss**: 두 픽셀 간의 예측값 차이 $f^d = |\text{ReLU}(f(p_1)) - \text{ReLU}(f(p_2))|$를 계산하여, GT가 같으면 $L^h(f^d)$, 다르면 $L^h(m_2 - f^d)$를 적용한다.

### 추론 절차 (Inference Pipeline)

1. **Label Prediction**: ResNet-50/101 기반의 Encoder-Decoder FCN이 단일 채널의 실수 값 맵을 출력한다.
2. **Discretization**: 출력된 실수 값 맵에 **Mean-shift segmentation** 알고리즘을 적용하여 연속적인 값을 이산적인 인스턴스 라벨로 변환한다.
3. **Verification**: 생성된 각 세그먼트의 Bounding Box에서 ROIAlign으로 특징을 추출하고, 7층의 작은 CNN을 통해 세만틱 클래스를 분류한다. 또한 **IoU head**를 통해 예측된 세그먼트의 품질을 예측하고, 분류 신뢰도와 IoU 예측값의 가중 합을 기준으로 가짜 양성(False Positive)을 제거한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL VOC 2012, PASCAL SBD, MS-COCO 2017.
- **지표**: mAP (average Precision), FPS (Frames Per Second).
- **비교 대상**: Mask R-CNN, YOLACT, SOLO, PolarMask 등 Search-based 및 Search-free 방법론.

### 정량적 결과

- **PASCAL VOC**: DVIS는 Search-free 방식인 SGN, Embedding보다 유의미하게 높은 성능을 보였으며, 특히 높은 IoU 임계값($0.7 \sim 0.9$)에서 더 정밀한 세그멘테이션 성능을 나타냈다.
- **MS-COCO**: Mask R-CNN과 같은 2-stage 방식에 근접한 성능을 보였으며, 특히 실시간 인스턴스 세그멘테이션 모델인 YOLACT보다 AP 성능이 높게 나타났다.
- **속도 및 효율성**: ResNet-50 기반 DVIS는 **38.0 FPS**를 기록하여 비교 대상 중 가장 빨랐다. 또한 후처리 단계로 넘어가는 후보군(Candidate)의 수가 평균 **5~15개**에 불과하여, 수백~수천 개의 Proposal을 처리하는 기존 방식보다 연산 효율이 압도적으로 높다.

### 정성적 결과

DVIS는 단일한 Global glance를 통해 객체를 파악하므로, 가려짐(Occlusion)으로 인해 분리된 객체의 파편들을 하나의 인스턴스로 묶는 능력이 뛰어났다. 또한, 학습 데이터에 없는 범주의 객체에 대해서도 '객체성(Objectness)'을 인식하여 세그멘테이션 하는 일반화 능력을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문의 가장 큰 강점은 인스턴스 세그멘테이션을 Proposal 탐색 문제가 아닌, **연속적인 최적화 문제**로 변환하여 해결했다는 점이다. 이를 통해 연산량을 획기적으로 줄이면서도, 변분법적 정규화를 통해 경계선의 정밀도를 높였다. 특히 Permutation-invariant loss는 라벨의 순서에 상관없이 인스턴스를 구분할 수 있게 하여 FCN이 직접 인스턴스 맵을 학습할 수 있는 길을 열었다.

### 한계 및 비판적 해석

- **작은 객체 탐지**: 논문에서도 언급되었듯, DVIS는 이미지 전체를 한 번에 보는 Top-down 방식이므로, 국소 영역을 세밀하게 탐색하는 Search-based 방식에 비해 **작은 객체(Small objects)**의 탐지 성능이 떨어진다. 이는 COCO 데이터셋의 $AP_S$ 지표에서 명확히 드러난다.
- **Crowded Scenes**: 객체가 매우 밀집된 환경에서는 인스턴스 맵이 모호해지는 경향이 있으며, 이는 특히 GT 라벨링이 일관되지 않은 경우(일부 객체만 라벨링 된 경우) 성능 저하로 이어진다.
- **후처리 의존성**: Mean-shift와 같은 클러스터링 기반 이산화 과정이 필수적이며, 이 과정의 bandwidth 파라미터 설정이 결과에 영향을 줄 수 있다.

## 📌 TL;DR

본 논문은 인스턴스 세그멘테이션을 Mumford-Shah 모델 기반의 변분 최적화 문제로 재정의하고, 이를 해결하는 **DVIS (Deep Variational Instance Segmentation)** 프레임워크를 제안한다. 순열 불변 손실 함수(Permutation-invariant loss)를 도입하여 FCN이 직접 실수 값의 인스턴스 맵을 예측하게 함으로써, 기존의 복잡한 Proposal 생성 과정 없이도 매우 빠른 속도(38 FPS)와 높은 정밀도를 달성하였다. 이 연구는 Proposal-free 인스턴스 세그멘테이션의 효율성을 입증하였으며, 특히 실시간성이 중요한 로보틱스나 영상 처리 분야에 적용될 가능성이 매우 높다.
