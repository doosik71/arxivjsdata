# Cataract-1K: Cataract Surgery Dataset for Scene Segmentation, Phase Recognition, and Irregularity Detection

Negin Ghamsarian, Yosuf El-Shabrawi, Sahar Nasirihaghighi, Doris Putzgruber-Adamitsch, Martin Zinkernagel, Sebastian Wolf, Klaus Schoeffmann, and Raphael Sznitman (2023)

## 🧩 Problem to Solve

본 연구는 백내장 수술(Cataract Surgery)의 컴퓨터 보조 중재(Computer-Assisted Interventions, CAI) 및 수술 후 비디오 분석을 위한 대규모 고품질 데이터셋의 부재 문제를 해결하고자 한다. 수술 단계 인식(Phase Recognition)과 장면 분할(Scene Segmentation)은 수술 계획, 숙련도 평가, 수술실 관리 및 결과 예측을 위한 Context-Aware Systems(CAS) 구축의 핵심 요소이다.

기존의 공개 데이터셋들은 특정 하위 작업(예: 도구 인식 또는 특정 해부학적 구조 분할)에만 치중되어 있거나, 규모가 매우 작아 딥러닝 모델의 일반화 성능을 높이는 데 한계가 있었다. 특히, 수술 중 발생하는 불규칙성(Irregularity)을 탐지하기 위한 종합적인 데이터셋이 부족하여, 수술 결과를 개선하고 사후 분석을 자동화하는 연구에 공백이 존재했다. 따라서 본 논문의 목표는 수술 단계, 해부학적 구조 및 도구의 세그멘테이션, 그리고 주요 수술 불규칙성을 모두 포함하는 대규모 다중 작업(Multi-task) 데이터셋인 'Cataract-1K'를 구축하고 그 효용성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 백내장 수술 비디오 분석을 위한 최대 규모의 데이터셋인 **Cataract-1K**를 제안한 것이다. 이 데이터셋의 설계 아이디어는 단순히 양을 늘리는 것이 아니라, 실제 임상에서 필요한 세 가지 핵심 관점(단계 인식, 장면 분할, 불규칙성 탐지)을 통합하여 제공함으로써 컴퓨터 비전 기반의 수술 분석 프레임워크를 가능하게 하는 데 있다.

구체적인 기여 사항은 다음과 같다:

- **대규모 데이터 수집**: 2021년부터 2023년 사이 오스트리아 Klinikum Klagenfurt에서 촬영된 1,000개의 백내장 수술 비디오를 확보하였다.
- **다중 작업 어노테이션**: 12가지 수술 단계에 대한 프레임 단위 라벨링, 3종의 해부학적 구조 및 9종의 수술 도구에 대한 픽셀 수준의 세그멘테이션 마스크를 제공한다.
- **불규칙성 서브셋 구축**: 동공 수축(Pupil contraction) 및 인공 수정체 회전(IOL rotation)이라는 두 가지 주요 수술 중 불규칙성 사례를 별도로 구축하여 특수 목적의 연구를 지원한다.
- **기술적 검증**: 최신(SOTA) 신경망 아키텍처를 벤치마킹하여 데이터셋의 품질을 검증하고, 서로 다른 데이터셋 간의 Domain Adaptation 필요성을 실험적으로 입증하였다.

## 📎 Related Works

논문에서는 기존의 백내장 수술 관련 데이터셋들이 특정 작업에 국한되어 있음을 지적한다. 예를 들어, 일부 데이터셋은 도구 인식(Instrument recognition)이나 특정 해부학적 구조의 분할만을 다루며, 인공 수정체(IOL) 불규칙성 탐지와 같은 특정 문제에 집중한 소규모 데이터셋만이 존재했다.

본 연구는 이러한 파편화된 기존 접근 방식과 달리, 하나의 통합된 데이터셋 내에서 단계 인식과 장면 분할을 동시에 수행할 수 있도록 설계되었다. 또한, 기존 연구들이 간과했던 수술 중 발생하는 비정상적인 상황(Irregularity)에 대한 데이터를 포함함으로써, 단순한 상황 인식을 넘어 수술 결과 예측 및 사후 분석으로 연구 범위를 확장했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Cataract-1K 데이터셋 구성

데이터셋은 크게 세 가지 부분으로 구성된다.

- **Phase Recognition Dataset**: 56개의 일반 수술 비디오에 대해 12가지 단계(Incision, Viscoelastic, Capsulorhexis, Hydrodissection, Phacoemulsification, Irrigation-Aspiration, Capsule Polishing, Lens Implantation, Lens Positioning, Viscoelastic-Suction, Anterior-Chamber Flushing, Tonifying/Antibiotics)와 Idle 단계를 정의하여 어노테이션하였다.
- **Semantic Segmentation Dataset**: 30개 비디오에서 추출한 2,256개 프레임에 대해 픽셀 수준의 라벨링을 수행하였다. 대상은 해부학적 구조(Iris, Pupil, IOL) 3종과 수술 도구(Slit/Incision Knife, Gauge, Spatula 등) 9종이다.
- **Irregularity Detection Dataset**: 수술 중 발생할 수 있는 동공 수축(Pupil reaction)과 인공 수정체 회전(IOL rotation) 사례를 포함한 소규모 서브셋이다.

### 2. 수술 단계 인식(Phase Recognition) 방법론

단계 인식을 위해 CNN-RNN 결합 프레임워크를 사용한다.

- **구조**: CNN(VGG16 또는 ResNet50)을 통해 각 프레임의 특징을 추출하고, 이를 RNN(LSTM, GRU, BiLSTM, BiGRU)에 입력하여 시계열적 특징을 캡처한다.
- **학습 전략**: 'One-versus-rest' 전략을 사용하여 각 단계별로 이진 분류를 수행하며, 데이터 불균형을 해소하기 위해 target 클래스와 rest 클래스의 클립 수를 동일하게 맞추는 랜덤 샘플링을 적용하였다.

### 3. 시맨틱 세그멘테이션(Semantic Segmentation) 방법론

다양한 SOTA 모델(UNet++, DeepPyramid, CE-Net 등)을 통해 검증을 수행하였으며, 학습 시 다음의 손실 함수를 사용하였다.

$$L = \lambda \times \text{BCE}(X_{true}, X_{pred}) - (1-\lambda) \times \left( \log_2 \frac{\sum X_{true} \odot X_{pred} + \sigma}{\sum X_{true} + \sum X_{pred} + \sigma} \right)$$

여기서 $\text{BCE}$는 Binary Cross Entropy를 의미하며, $\odot$은 Hadamard product(원소별 곱)이다. $\lambda$는 $0.8$로 설정되었으며, $\sigma$는 분모가 0이 되는 것을 방지하고 오버피팅을 막기 위한 Laplacian smoothing factor($\sigma=1$)이다.

## 📊 Results

### 1. 수술 단계 인식 결과

- **모델 성능**: 양방향 RNN(BiLSTM, BiGRU)을 사용했을 때 정확도(Accuracy)와 F1-Score가 일관되게 향상되었다.
- **단계별 난이도**: Phacoemulsification 단계가 가장 높은 정확도를 보였는데, 이는 해당 단계에서 사용하는 도구가 매우 독특하고 동공의 텍스처가 뚜렷하기 때문이다. 반면, Viscoelastic 및 AC Flushing 단계는 시각적 특징이 유사하여 가장 낮은 성능을 기록하였다.

### 2. 시맨틱 세그멘테이션 결과

- **객체별 성능**: 해부학적 구조 분할이 도구 분할보다 훨씬 쉬운 작업임이 확인되었다. 특히 동공(Pupil)은 경계가 뚜렷하여 성능이 가장 좋았으나, 인공 수정체(Lens)는 투명한 특성상 성능이 상대적으로 낮았다.
- **최적 모델**: VGG16 백본을 사용하는 **DeepPyramid** 네트워크가 모든 클래스에서 가장 최적의 성능을 보였다.

### 3. Cross-domain 실험 결과

Cataract-1K에서 학습한 모델을 CaDIS 데이터셋에 적용했을 때, binary instrument segmentation의 Dice 계수가 $77\%$에서 $67\%$로 크게 하락하였다. 이는 두 데이터셋 간의 상당한 Domain Shift가 존재함을 의미하며, 향후 Semi-supervised learning이나 Domain Adaptation 기술의 필요성을 시사한다.

## 🧠 Insights & Discussion

본 연구를 통해 백내장 수술 비디오 분석에서 발생하는 몇 가지 핵심적인 기술적 난제가 확인되었다. 첫째, 수술 도구의 경우 반사(Reflection), 모션 블러(Motion blur), 그리고 다른 도구에 의한 가려짐(Occlusion) 현상으로 인해 세그멘테이션 성능이 저하된다. 둘째, 인공 수정체의 투명성과 변형 가능성은 픽셀 단위의 정밀한 분할을 어렵게 만드는 요인이다.

또한, 데이터셋 간의 도메인 격차가 크다는 점은 매우 중요한 발견이다. 이는 특정 병원이나 장비에서 수집된 데이터로 학습된 모델이 다른 환경에서도 작동하기 위해서는 단순한 데이터 증강을 넘어선 도메인 적응 전략이 필수적임을 보여준다. 본 논문은 대규모 데이터셋을 제공함으로써 이러한 문제를 해결하기 위한 기초 토대를 마련하였다는 점에서 강점이 있다.

## 📌 TL;DR

본 논문은 백내장 수술의 단계 인식, 장면 분할, 불규칙성 탐지를 위해 구축된 최대 규모의 데이터셋 **Cataract-1K**(비디오 1,000개)를 제안한다. 실험을 통해 양방향 RNN 기반의 단계 인식과 DeepPyramid 기반의 세그멘테이션 성능을 검증하였으며, 특히 서로 다른 데이터셋 간의 큰 도메인 차이를 확인하여 향후 Domain Adaptation 연구의 필요성을 제시하였다. 이 연구는 수술 자동 분석 및 숙련도 평가 시스템의 실전 적용을 위한 핵심적인 데이터 인프라를 제공한다.
