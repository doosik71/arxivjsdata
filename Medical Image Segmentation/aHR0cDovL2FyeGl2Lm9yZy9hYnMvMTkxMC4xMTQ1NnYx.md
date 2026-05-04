# Unified Multi-scale Feature Abstraction for Medical Image Segmentation

Xi Fang, Bo Du, Sheng Xu, Bradford J. Wood, and Pingkun Yan (2019)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석의 핵심 요소인 자동 의료 영상 분할(Automatic Medical Image Segmentation) 문제를 다룬다. 특히 간(Liver)의 위치를 찾고 분할하는 작업은 간암의 진단과 치료에 매우 중요하지만, 정밀한 분할을 위해서는 다양한 크기의 특징(Multi-scale features)을 효과적으로 추출하고 통합하는 것이 필수적이다.

기존의 FCN(Fully Convolutional Network)이나 U-Net 기반의 모델들은 주로 네트워크 구조 최적화(예: ResNet, DenseNet 도입)에 집중해 왔으나, 단일 스케일의 특징 추출 방식만으로는 의료 영상 내의 복잡한 계층적 정보를 완전히 활용하는 데 한계가 있다. 따라서 본 연구의 목표는 멀티 스케일 입력을 활용하고 이를 효율적으로 결합하는 새로운 네트워크 아키텍처를 설계하여 분할 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **MIMO-FAN**이라는 새로운 멀티 스케일 네트워크 아키텍처를 제안하는 것이다. 이 모델의 중심적인 직관은 네트워크의 시작 단계부터 다양한 스케일의 입력을 처리하고, 전체 네트워크 깊이에 걸쳐 이들 스케일 간의 특징을 밀집하게 연결하여 글로벌 문맥(Global context)과 로컬 세부 정보(Local detail)를 동시에 보존하고 통합하는 것이다.

이를 위해 저자들은 다음과 같은 세 가지 핵심 메커니즘을 도입하였다:
1. **Dense Cross-scale Connections (DCC)**: 동일한 깊이(Depth)에 있는 서로 다른 스케일의 특징 맵들을 밀집하게 연결하여 계층적 정보를 강화한다.
2. **Deep Pyramid Supervision (DPS)**: 디코딩 단계에서 다양한 스케일의 출력물을 생성하고 이를 각각 감독함으로써 기울기 소실(Gradient vanishing) 문제를 완화하고 정교한 마스크를 생성한다.
3. **Scale Fusing (SF)**: 최종적으로 서로 다른 스케일에서 생성된 확률 맵 중 가장 큰 두 개의 맵을 융합하여 신뢰도 높은 최종 분할 결과를 얻는다.

## 📎 Related Works

기존의 의료 영상 분할 연구들은 주로 U-Net 및 그 변형 모델들을 사용해 왔으며, 최근에는 ResNet이나 DenseNet과 같은 최신 CNN 구조를 통합하여 특징 추상화 능력을 높이려는 시도가 많았다.

특히 간 CT 영상 분할의 경우, 많은 최신 기법들이 **2단계 접근 방식(Two-step approach)**을 취한다. 먼저 거친 분할(Coarse segmentation)을 통해 간의 대략적인 위치를 찾고, 이후 정밀 분할(Fine segmentation) 단계를 통해 최종 결과를 얻는 방식이다. 또한, 성능 향상을 위해 2D 특징과 3D 특징을 함께 사용하는 하이브리드 모델(예: H-DenseUNet)들이 제안되었다. 그러나 이러한 방식들은 계산 비용이 매우 높으며, 훈련 시간이 오래 걸린다는 단점이 있다. MIMO-FAN은 이러한 복잡한 다단계나 3D 접근 방식 없이, 단일 2D 네트워크만으로 경쟁력 있는 성능을 내는 것을 목표로 하여 차별성을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조
MIMO-FAN은 입력 이미지에 대해 **Spatial Pyramid Pooling**을 적용하여 서로 다른 스케일의 입력들을 생성하는 것으로 시작한다. 이후 공유 커널(Shared kernels)을 가진 첫 번째 컨볼루션 블록을 통과하며 전체 장면을 해석하는 이미지 수준의 문맥 특징을 추출한다.

### 주요 구성 요소
1. **Dense Cross-scale Connections (DCC)**
   DCC는 네트워크의 각 레벨에서 서로 다른 스케일의 특징 맵들을 융합하는 모듈이다.
   - **작동 방식**: 작은 스케일의 특징 맵은 더 넓은 글로벌 문맥 정보를 포함하고, 큰 스케일은 로컬 세부 정보를 포함한다. DCC는 동일한 깊이의 특징 맵들 사이에 밀집 연결(Dense connection)을 추가하여 이를 상호 보완한다.
   - **흐름**: 인코더(Encoder) 부분에서는 상위에서 하위로(Top-down) 연결하여 특징을 결합하고, 디코더(Decoder) 부분에서는 하위에서 상위로(Bottom-up) 연결하여 고수준 특징을 점진적으로 복원한다.

2. **Deep Pyramid Supervision (DPS)**
   다양한 스케일에서 효율적인 특징 추상화가 이루어지도록 하기 위해, 디코딩 단계의 여러 스케일 출력물에 대해 직접적으로 감독(Supervision)을 가한다.
   - **라벨 생성**: Ground Truth(GT) 분할 맵에 Spatial Pyramid Pooling을 적용하여 각 출력 스케일에 맞는 라벨을 생성한다.
   - **손실 함수**: 가중치 교차 엔트로피(Weighted Cross Entropy)를 사용하여 각 스케일의 예측값과 GT 간의 오차를 계산한다.

   $$L = -\frac{1}{S} \sum_{s=1}^{S} \frac{1}{N_s} \sum_{i=1}^{N_s} \sum_{c=0}^{C} w_{c} y_{c,i,s} \log p_{c,i,s}$$

   여기서 $S$는 총 스케일 수(본 논문에서는 5), $p_{c,i,s}$는 스케일 $s$에서 복셀 $i$가 클래스 $c$에 속할 확률, $y_{c,i,s}$는 정답 라벨, $w_c$는 클래스별 가중치이다. (배경: 0.2, 간: 1.2로 설정)

3. **Scale Fusing (SF)**
   DPS를 통해 생성된 여러 스케일의 확률 맵 중 가장 크기가 큰 두 개의 맵을 융합하여 최종 세그멘테이션 결과를 도출한다.

## 📊 Results

### 실험 설정
- **데이터셋**: LiTS (Liver Tumor Segmentation Challenge) 데이터셋 (훈련 131개, 테스트 70개).
- **전처리**: $512 \times 512$ 영상을 $256 \times 256$으로 리사이징.
- **검증 및 평가**: 5-fold 교차 검증을 수행하고, 테스트 셋 제출 시에는 5개 모델의 결과에 대해 다수결 투표(Majority voting)를 적용하였다.

### 정량적 결과 비교
MIMO-FAN은 기존의 고성능 모델들과 비교했을 때 매우 경쟁력 있는 성능을 보였다.

- **정확도**: Global Dice 점수 기준, 최상위 모델(DeepX 등)과의 차이가 $0.5\%$ 미만으로 매우 근소하였다.
- **효율성**: 기존의 H-DenseUNet 등이 2D DenseUNet 훈련에 21시간, 파인튜닝에 9시간(총 30시간)을 소요한 반면, MIMO-FAN은 단일 Titan Xp GPU에서 **단 3시간 만에** 훈련을 완료하였다. 또한, 2단계 과정 없이 한 번의 단계(One-step)로 분할을 수행한다.

### 절제 연구 (Ablation Study)
U-Net, ResU-Net, DenseU-Net과 비교하여 DCC와 DPS의 효과를 검증하였다.
- **결과**: 기본 U-Net 계열 모델들보다 MIMO-FAN이 통계적으로 유의미하게 높은 성능을 보였다 ($p$-value $\le 0.025$).
- **구성 요소별 기여**: `DCC` $\rightarrow$ `DCC+DPS` $\rightarrow$ `DCC+DPS+SF` 순으로 Dice 점수가 점진적으로 상승하였으며, 모든 구성 요소가 포함되었을 때 가장 높은 Mean Dice ($\approx 95.7\%$)를 기록하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **연산 효율성과 정확성 사이의 균형**을 매우 잘 맞췄다는 점이다. 기존의 고성능 모델들이 3D 컨볼루션을 사용하거나 복잡한 2단계 파이프라인을 구축하여 연산 비용을 높였던 반면, MIMO-FAN은 2D 네트워크 내에서 멀티 스케일 특징을 효율적으로 융합하는 구조(DCC)와 감독 체계(DPS)를 도입함으로써 유사한 성능을 훨씬 빠르게 달성하였다.

특히, 단순한 스킵 연결(Skip connection)을 넘어 서로 다른 스케일 간의 특징을 직접적으로 교환하는 DCC 구조는 의료 영상처럼 객체의 크기 변화가 심한 데이터에서 글로벌 문맥과 로컬 세부 사항을 동시에 포착하는 데 매우 유효한 전략임을 입증하였다.

다만, 본 연구는 주로 간 분할이라는 특정 작업에 집중되어 있으며, 다른 장기나 더 복잡한 병변에 대해서도 동일한 효율성이 유지될지는 추가적인 검증이 필요하다. 또한 2D 기반 모델이기에 슬라이스 간의 연속성(Inter-slice continuity)을 완전히 활용하지 못한다는 3D 모델 대비 근본적인 한계가 존재할 수 있다.

## 📌 TL;DR

MIMO-FAN은 멀티 스케일 입력과 출력을 단일 2D 네트워크 내에서 통합 관리하는 구조로, **Dense Cross-scale Connections(DCC)**와 **Deep Pyramid Supervision(DPS)**를 통해 간 분할 성능을 극대화하였다. 기존의 무거운 3D 모델이나 2단계 분할 방식보다 훨씬 빠른 훈련 및 추론 속도를 가지면서도 이에 근접하는 높은 정확도를 달성하여, 의료 영상 분할의 효율적인 새로운 방향성을 제시하였다.