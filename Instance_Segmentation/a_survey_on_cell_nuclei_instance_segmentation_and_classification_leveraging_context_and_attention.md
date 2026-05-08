# A Survey on Cell Nuclei Instance Segmentation and Classification: Leveraging Context and Attention

João D. Nunes, Diana Montezuma, Domingos Oliveira, Tania Pereira, Jaime S. Cardoso (2024)

## 🧩 Problem to Solve

본 논문은 Haematoxylin and Eosin (H&E) 염색된 Whole Slide Images (WSIs)에서 세포 핵의 인스턴스 분할(Instance Segmentation) 및 분류(Classification)를 자동화하는 문제를 다룬다. 세포 핵의 형태학적 특징과 바이오마커는 종양 미세환경에 대한 중요한 통찰을 제공하며, 암의 진단 및 예후 판정에 필수적이다. 그러나 기가픽셀 단위의 거대한 WSI 이미지에서 수동으로 핵을 주석(Annotation)하는 작업은 막대한 비용과 시간이 소모된다.

자동화 알고리즘의 도입이 절실함에도 불구하고, 세포 핵의 형태 및 색상 특징의 높은 클래스 내/간 변동성(Intra- and Inter-class variability)과 H&E 염색 과정에서 발생하는 아티팩트(Artefact)로 인해 기존의 알고리즘들은 임상 수준의 성능을 달성하는 데 어려움을 겪고 있다. 따라서 본 연구의 목표는 인공신경망(ANN)에 '문맥(Context)'과 '주의(Attention)'라는 귀납적 편향(Inductive Bias)을 도입함으로써, 세포 핵 분할 및 분류의 성능과 일반화 능력을 향상시킬 수 있는지 분석하고 이를 위한 체계적인 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **문맥 및 주의 메커니즘의 체계적 분류**: 컴퓨터 비전 일반 및 의료 영상 분야에서 사용되는 Attention과 Context의 정의를 정립하고, 이를 수준(Level)과 유형(Type)에 따라 체계적으로 분류한 택소노미(Taxonomy)를 제시하였다.
2. **H&E 염색 이미지 특화 서베이**: 2019년부터 2023년까지 발행된 Q1 저널 및 주요 컨퍼런스 논문 중, H&E 염색 이미지의 세포 핵 인스턴스 분할 및 분류에 문맥과 주의 메커니즘을 적용한 44편의 논문을 심층 분석하였다.
3. **실증적 사례 연구(Case Study)**: 일반적인 인스턴스 분할 모델인 Mask-RCNN과 핵 분할 특화 모델인 HoVer-Net에 Squeeze-and-Excitation(SE) 블록과 Semantic Relation Module(SRM)을 추가하여, 이러한 단순한 모듈 확장이 실제 성능 및 일반화에 미치는 영향을 정량적으로 평가하였다.
4. **미래 연구 방향 제시**: Causal Representation Learning, Graph Neural Networks (GNN), Multimodal ML 등 향후 세포 핵 분석의 성능을 높이기 위한 최신 기술적 대안을 논의하였다.

## 📎 Related Works

기존의 연구들은 주로 U-Net, Mask-RCNN, HoVer-Net과 같은 딥러닝 아키텍처를 이용한 핵 분할 성능 향상에 집중해 왔다. 최근에는 Vision Transformer (ViT) 기반의 모델들이 일반 컴퓨터 비전 분야에서 압도적인 성능을 보이며 의료 영상 분야에도 도입되고 있다.

그러나 본 논문은 기존 서베이 논문들이 단순히 모델의 성능 수치(Benchmark) 비교에 치중한 것과 달리, 병리학자가 WSI를 분석할 때 사용하는 '다양한 수준의 문맥 정보'와 '특정 관심 영역(RoI)에 대한 주의'라는 도메인 지식을 알고리즘 설계에 어떻게 반영할 것인가에 집중한다. 특히, 단순한 성능 향상을 넘어 도메인 일반화(Domain Generalization)와 데이터 효율성 관점에서 문맥-주의 메커니즘의 효용성을 분석한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 문맥(Context)과 주의(Attention)의 정의 및 분류

논문은 Attention을 다음과 같은 일반식으로 정의한다.
$$A = f(g(x), x)$$
여기서 $g(x)$는 주의 메커니즘을, $f(g(x), x)$는 주의 결과에 기반하여 입력 $x$를 처리하는 과정을 의미한다. Attention은 채널(Channel), 공간(Spatial), 시간(Temporal), 브랜치(Branch) 주의로 분류된다.

문맥(Context)은 다음과 같은 수준(Level)과 유형(Type)으로 세분화하여 정의한다.

- **수준(Levels)**: Prior knowledge, Global, Local, Long Range, Multiscale context.
- **유형(Types)**: Spatial, Temporal, Semantic context.

### 2. 분석 대상 모델 및 아키텍처

본 연구에서는 다음과 같은 핵심 모듈을 분석하고 실험에 적용하였다.

- **Squeeze-and-Excitation (SE) Block**: 전역 평균 풀링(Global Average Pooling)을 통해 채널 간의 의존성을 모델링하고, 중요도가 낮은 채널은 억제하고 높은 채널은 강화하는 메커니즘이다.
- **Convolutional Block Attention Module (CBAM)**: 채널 주의와 공간 주의를 순차적으로 적용하여 '무엇이(What)' 그리고 '어디가(Where)' 중요한지를 동시에 학습한다.
- **Semantic Relation Module (SRM)**: 객체 간의 관계를 학습하기 위해 Transformer Encoder를 도입하여, 개별 인스턴스의 특징뿐만 아니라 인스턴스 간의 상호작용과 장거리 의존성(Long-range dependency)을 인코딩한다.

### 3. 사례 연구(Case Study) 실험 설계

저자들은 Mask-RCNN과 HoVer-Net을 베이스라인으로 설정하고 다음과 같이 확장하였다.

- **Mask-RCNN 변형**:
  - `Mask-RCNN-ResNet50-SE-FPN`: FPN의 모든 lateral connection 이전에 SE-block을 추가하여 특징 맵을 정제한다.
  - `Mask-RCNN-ResNet50-FPN-SRM`: MLP-Head의 두 번째 FC 레이어를 Transformer Encoder로 교체하여 객체 간 관계를 학습한다.
- **HoVer-Net 변형**:
  - `SE-HoVer-Net`: ResNet-50 백본의 모든 블록에 SE-block을 추가한다.
  - `SRM-HoVer-Net`: 백본의 최하단 레이어에 SRM을 추가한다.

**학습 절차 및 손실 함수**:
클래스 불균형 문제를 해결하기 위해 Focal Loss를 사용하였으며, 다음과 같이 타일 내 샘플의 유효 개수($n$)를 고려하여 가중치를 부여하였다.
$$L(p_t) = -\frac{1-\beta}{1-\beta^n}(1-p_t)^\gamma \log(p_t)$$
여기서 $\beta, \gamma$는 하이퍼파라미터이며, $p_t$는 예측 확률이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Lizard 데이터셋 (6개 기관의 데이터, 6종의 핵 타입 포함).
- **검증 방법**: Leave-one-domain-out validation을 통해 모델의 일반화 능력을 평가하였다.
- **평가 지표**: Dice Coefficient (DC), Aggregated Jaccard Index (AJI), Panoptic Quality (PQ) 등을 사용하였다.

### 2. 주요 결과

- **HoVer-Net 실험**: SE-block과 SRM을 추가했을 때, 베이스라인 대비 $\text{mPQ}^+$ 수치가 각각 $1.83\%$, $2.93\%$ 향상되는 경향을 보였다. 이는 도메인 특화 모델에서는 문맥-주의 메커니즘이 긍정적인 영향을 줄 수 있음을 시사한다.
- **Mask-RCNN 실험**: 결과가 매우 불안정하였다. 일부 설정에서는 성능이 향상되었으나, 다른 설정에서는 오히려 저하되는 양상을 보였으며, 통계적 유의성 검정(T-test) 결과에서도 일관성이 없었다.
- **일반화 능력**: 단순한 모듈 추가만으로는 서로 다른 데이터 소스(Domain) 간의 성능 격차를 획기적으로 줄이는 일반화 효과를 얻기 어려웠다.

## 🧠 Insights & Discussion

### 1. 도메인 지식의 알고리즘 반영 난이도

병리학자가 직관적으로 사용하는 다층적 문맥 정보를 인공신경망의 아키텍처로 변환하는 것은 매우 어려운 작업이다. 본 실험에서 보았듯, 단순히 범용적인 Attention 모듈(SE-block 등)을 추가하는 것만으로는 유의미한 성능 향상을 보장할 수 없다. 이는 모델이 단순히 파라미터 수만 늘어나 과적합(Overfitting)될 위험이 있기 때문이다.

### 2. ConvNet의 내재적 문맥 학습 능력

저자들은 최신 ConvNet(특히 FPN 구조를 가진 모델)들이 이미 학습 데이터로부터 상당 수준의 문맥 정보를 내재적으로 학습하고 있을 가능성을 제기한다. 따라서 명시적인 Attention 레이어를 추가하는 것이 추가적인 정보 이득을 주지 못할 수 있다는 가설을 세웠다.

### 3. 데이터셋의 한계

현재 공개된 데이터셋들은 핵의 경계와 타입에 대한 주석만 제공할 뿐, 해당 핵이 속한 조직(Tissue)이나 세포 공동체(Cellular community)와 같은 '문맥적 주석'이 부족하다. 이러한 고수준의 문맥 정보가 포함된 데이터셋이 구축된다면, 더 정교한 지도 학습(Supervised Learning)이 가능할 것이다.

### 4. 미래 방향성에 대한 제언

단순한 아키텍처 확장을 넘어 다음과 같은 접근 방식이 필요함을 역설한다.

- **Causal Representation Learning**: 단순 통계적 상관관계가 아닌 인과 구조를 학습하여 도메인 시프트에 강건한 모델을 구축해야 한다.
- **Graph Neural Networks (GNN)**: 세포 핵 간의 관계를 그래프 구조로 정의하여 비유클리드 공간에서의 상호작용을 모델링하는 것이 효과적일 수 있다.
- **Multimodal ML**: 이미지뿐만 아니라 임상 보고서, 분자 프로파일링 데이터를 통합하여 더 넓은 범위의 세만틱 문맥을 확보해야 한다.

## 📌 TL;DR

본 논문은 H&E 염색 이미지의 세포 핵 분할 및 분류 성능을 높이기 위해 '문맥(Context)'과 '주의(Attention)' 메커니즘을 적용한 연구들을 체계적으로 분석한 서베이 논문이다. 분석 결과, 단순한 Attention 모듈의 추가가 항상 성능 향상으로 이어지지는 않으며, 특히 일반 모델(Mask-RCNN)보다 도메인 특화 모델(HoVer-Net)에서 더 효과적임을 확인하였다. 결론적으로, 병리학적 도메인 지식을 알고리즘으로 구현하기 위해서는 단순한 모듈 결합이 아닌, 인과 추론이나 그래프 구조 도입과 같은 정교하고 맞춤화된 설계가 필요함을 강조한다.
