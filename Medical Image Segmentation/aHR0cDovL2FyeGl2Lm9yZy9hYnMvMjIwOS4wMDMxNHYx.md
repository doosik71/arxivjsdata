# Self-Supervised Pretraining for 2D Medical Image Segmentation

András Kalapos and Bálint Gyires-Tóth (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 정답 라벨링(Annotation)에 소요되는 높은 비용과 전문 지식의 필요성 문제를 해결하고자 한다. 의료 영상 데이터는 수집하더라도 라벨이 없는 상태로 존재하는 경우가 많으며, 특히 희귀 질환의 경우 라벨링된 데이터를 대량으로 확보하는 것이 매우 어렵다.

기존에는 ImageNet과 같은 자연 이미지 데이터셋으로 학습된 모델을 전이 학습(Transfer Learning)하는 방식이 주로 사용되었으나, 자연 이미지와 의료 영상 간의 데이터 분포 및 특징의 차이로 인해 효율성이 떨어질 수 있다는 문제가 있다. 따라서 본 연구의 목표는 자기지도 학습(Self-Supervised Learning, SSL)을 이용한 사전 학습(Pretraining) 전략이 하위 작업인 의료 영상 분할 모델의 수렴 속도와 데이터 효율성에 어떠한 영향을 미치는지 분석하고, 최적의 사전 학습 가이드라인을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 사전 학습의 범위와 순서에 따라 세 가지 파이프라인을 정의하고, 이를 의료 영상 분할 작업에 적용하여 그 효과를 정량적으로 분석한 점이다.

1. **Generalist Pretraining**: 자연 이미지(ImageNet)를 이용한 지도 학습 또는 자기지도 학습 기반의 사전 학습이다.
2. **Specialist Pretraining**: 타겟 도메인(본 연구에서는 ACDC 심장 MRI 데이터셋)의 라벨 없는 데이터를 이용한 자기지도 학습 기반의 사전 학습이다.
3. **Hierarchical Pretraining**: Generalist 단계 이후 Specialist 단계를 순차적으로 수행하는 계층적 사전 학습 방식이다.

특히, 자기지도 학습만으로 구성된 계층적 사전 학습(BYOL ImageNet $\rightarrow$ BYOL ACDC)이 하위 작업의 수렴 속도와 안정성 면에서 가장 우수함을 입증하였다.

## 📎 Related Works

논문에서는 자기지도 학습(SSL)이 라벨 없는 데이터로부터 유용한 표현(Representation)을 학습하여 하위 작업의 성능을 높이는 방법론임을 설명한다. 초기 SSL은 지그소 퍼즐 맞추기(Jigsaw puzzle), 이미지 패치 위치 예측, 색상화(Colorization) 등 휴리스틱한 Pretext Task를 사용하였으나, 최근에는 인스턴스 판별(Instance Discrimination) 방식이 주류를 이루고 있다. 이는 동일한 이미지의 서로 다른 증강(Augmentation) 뷰들이 유사한 잠재 벡터(Latent vector)를 갖도록 학습하는 방식이다.

의료 영상 분야에서는 자연 이미지와 의료 영상의 도메인 차이(Domain Gap)가 존재함에도 불구하고 ImageNet 기반 전이 학습이 널리 쓰이고 있다. 최근 SimCLR나 MoCo와 같은 SSL 기법들이 의료 영상 분류 작업에서 지도 학습 기반의 ImageNet 사전 학습보다 1-5% 더 높은 정확도를 보였다는 연구가 보고되었으나, 의료 영상 분할(Segmentation) 작업에 이러한 SSL 사전 학습을 적용하여 데이터 효율성을 분석한 연구는 상대적으로 부족한 실정이다.

## 🛠️ Methodology

### 전체 파이프라인 구조
학습 과정은 사전 학습 단계(1~2단계)와 하위 작업인 지도 학습 기반의 분할(Downstream Supervised Segmentation) 단계로 구성된다. 각 단계 사이에는 인코더(Encoder)의 가중치를 전이하며, 새롭게 추가되는 레이어(Projector, Predictor, Decoder 등)는 Kaiming-uniform 초기화를 통해 랜덤하게 초기화한다.

### BYOL (Bootstrap Your Own Latent) 알고리즘
본 논문은 표현 붕괴(Representation Collapse)를 방지하기 위해 비대칭 구조를 사용하는 BYOL 기법을 채택하였다. 시스템은 크게 Online 네트워크와 Target 네트워크로 나뉜다.

1. **Online Network**: 인코더 $f_\theta$, 프로젝터 $g_\theta$, 프리딕터 $q_\theta$로 구성된다. 입력 이미지 $x_1$에 대해 최종 출력은 $q_\theta(g_\theta(f_\theta(x_1)))$이 된다.
2. **Target Network**: 인코더 $f_\xi$와 프로젝터 $g_\xi$로 구성된다. 입력 이미지 $x_2$에 대해 출력은 $g_\xi(f_\xi(x_2))$이다.

Target 네트워크의 가중치 $\xi$는 Online 네트워크의 가중치 $\theta$를 지수 이동 평균(Exponential Moving Average, EMA) 방식으로 업데이트하며, 그래디언트 전파는 이루어지지 않는다. 학습 목표는 두 뷰의 출력 간의 평균 제곱 오차(Mean Squared Error, MSE)를 최소화하는 것이다.

$$ \text{Loss} = \| q_\theta(z_1) - z_2 \|^2 $$

여기서 $z_1$과 $z_2$는 각각 Online과 Target 네트워크의 프로젝터 출력값이다.

### 데이터셋 및 모델 아키텍처
- **데이터셋**: ACDC(Automated Cardiac Diagnosis Challenge) 데이터셋을 사용하며, 100명의 환자로부터 얻은 약 25,000장의 2D MRI 슬라이스가 포함되어 있다. 이 중 1,900장만 라벨이 존재하며, 나머지는 SSL 사전 학습에 활용한다.
- **모델 아키텍처**: 인코더로는 ResNet-50을 사용하고, 전체 구조는 U-Net을 채택하였다. ImageNet 사전 학습 모델의 경우 3채널 RGB 입력을 처리하므로, 단일 채널 MRI 영상을 위해 첫 번째 레이어의 가중치를 깊이 방향으로 합산(Depth-wise summation)하여 조정하였다.

## 📊 Results

### 수렴 속도 및 안정성
실험 결과, 계층적 사전 학습(Hierarchical Pretraining) 방식이 가장 빠른 수렴 속도를 보였다. 특히 **BYOL ImageNet + BYOL ACDC** 조합은 일반적인 ImageNet 사전 학습 모델보다 하위 작업 수렴 속도가 4~5배 더 빨랐으며, 학습 곡선이 매우 안정적이었다. 반면, 지도 학습 기반의 ImageNet 사전 학습 후 SSL을 적용한 경우는 최종 Jaccard Index(IoU)가 다른 방법론보다 유의미하게 낮게 나타났다.

### 데이터 효율성 (Data-efficiency)
라벨링된 데이터의 양을 1개부터 전체 데이터셋까지 변화시키며 테스트한 결과, 예상과 달리 매우 적은 양의 데이터(약 2% 미만, 30장 이하) 환경에서는 Generalist 사전 학습(특히 Supervised ImageNet)이 계층적 사전 학습보다 더 낮은 테스트 오차를 보였다.

데이터 양에 따른 오차 감소는 전형적인 멱법칙(Power-law) 스케일링을 따르며, 약 30~75장의 슬라이스를 기점으로 오차 감소 폭이 줄어드는 전이 영역(Transition region)이 관찰되었다. 이는 30~75장 정도의 라벨만으로도 최소 오차에 근접한 성능을 낼 수 있음을 시사한다.

### 도메인 특정 사전 학습의 에폭 수
계층적 사전 학습 파이프라인에서 도메인 특정(ACDC) SSL 단계를 몇 에폭 수행해야 하는지 실험한 결과, 단 3-4 에폭만으로도 앞서 언급한 빠른 수렴 속도와 안정적인 학습 곡선을 얻을 수 있음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 의료 영상 분할에서 SSL 사전 학습의 실용적인 가이드라인을 제시한다. 가장 큰 강점은 **자기지도 학습 기반의 계층적 전이 학습이 하위 작업의 학습 시간을 획기적으로 단축**시킬 수 있다는 점을 정량적으로 증명한 것이다. 이는 계산 자원이 제한적이거나 빠른 프로토타이핑이 필요한 환경에서 매우 유용한 전략이 될 수 있다.

그러나 데이터 효율성 측면에서의 결과는 흥미로운 시사점을 준다. 극단적으로 라벨 데이터가 부족한 상황에서는 도메인 특화 SSL보다 일반적인 이미지의 풍부한 특징을 학습한 지도 학습 기반의 ImageNet 모델이 더 강건한 성능을 보였다. 이는 도메인 전이가 일어나더라도 일반적인 시각적 특징(엣지, 질감 등)이 매우 적은 양의 데이터로 세밀한 튜닝을 수행할 때 더 유리할 수 있음을 의미한다.

결론적으로, 학습 속도와 안정성을 중시한다면 SSL 기반의 계층적 사전 학습을, 극소량의 데이터로 최대 성능을 짜내야 한다면 기존의 Supervised ImageNet 사전 학습을 고려하는 것이 합리적이다.

## 📌 TL;DR

이 논문은 의료 영상 분할을 위해 **자연 이미지 $\rightarrow$ 의료 영상** 순으로 이어지는 **계층적 자기지도 학습(Hierarchical SSL)**이 하위 작업의 수렴 속도를 4-5배 높인다는 것을 입증하였다. 또한, cardiac MRI 분할 작업에서 약 30~75장의 라벨링된 데이터만으로도 충분한 성능을 낼 수 있음을 보여주어, 의료 분야의 라벨링 비용 문제를 완화할 수 있는 실질적인 방안을 제시하였다.