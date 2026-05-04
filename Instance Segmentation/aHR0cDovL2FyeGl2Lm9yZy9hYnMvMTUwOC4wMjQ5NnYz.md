# A practical guide to CNNs and Fisher Vectors for image instance retrieval

Vijay Chandrasekhar, Jie Lin, Olivier Morère, Hanlin Goh, Antoine Veillard (2015)

## 🧩 Problem to Solve

본 논문은 이미지 인스턴스 검색(Image Instance Retrieval) 문제에서 가장 효과적인 글로벌 이미지 기술자(Global Image Descriptor)가 무엇인지 분석하는 것을 목표로 한다. 이미지 인스턴스 검색은 쿼리 이미지와 동일한 객체나 장면을 포함하고 있는 이미지를 데이터베이스에서 찾아내는 작업이다. 

최근 컴퓨터 비전 분야에서는 딥러닝 기반의 Convolutional Neural Networks(CNN)가 지배적인 접근 방식으로 자리 잡았으며, 기존의 최첨단(State-of-the-art) 기술이었던 Fisher Vectors(FV)를 빠르게 대체하고 있다. 그러나 CNN이 이미지 분류(Classification)에서는 압도적인 성능을 보이지만, 인스턴스 검색이라는 특수한 맥락에서 FV보다 항상 우위에 있는지는 명확히 밝혀지지 않았다. 특히 검색 작업에서 중요한 요소인 기하학적 변환(회전, 크기 변화)에 대한 강건성(Robustness) 측면에서 두 방식의 동작 특성이 크게 다르다. 따라서 본 연구는 CNN과 FV를 체계적으로 비교 분석하여, 특정 검색 문제에 가장 적합한 기술자를 선택할 수 있는 실무적인 가이드라인을 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 CNN과 FV 기술자에 대한 광범위하고 체계적인 벤치마크 분석을 통해 다음과 같은 인사이트를 제공한 점이다.

1. **기술자별 최적 설정(Best Practices) 도출**: CNN의 경우 이미지 크롭 전략, 추출 레이어, 네트워크 깊이, 학습 데이터의 영향을 분석하였으며, FV의 경우 관심 지점(Interest point) 검출 방식(Sparse vs Dense)에 따른 성능 차이를 규명하였다.
2. **상호 보완적 결합(Fusion) 제안**: CNN과 FV가 서로 다른 특성을 가지고 있음을 확인하고, 두 기술자를 단순 결합하는 것만으로도 단일 기술자를 사용할 때보다 성능이 유의미하게 향상됨을 보였다.
3. **기하학적 변환 강건성 분석 및 해결책 제시**: CNN이 회전에 매우 취약하고 크기 변화에는 비교적 강건하다는 점을 밝혀냈으며, 이를 해결하기 위해 데이터베이스 측에서 풀링(Pooling)을 수행하는 기법(Max-pooling, Min-dist PWL)을 제안하였다.

## 📎 Related Works

기존의 이미지 검색 연구는 크게 수작업으로 설계된 특징(Hand-crafted features) 기반의 FV와 학습 기반의 CNN 표현력으로 나뉜다.

- **Fisher Vectors (FV)**: 지역 특징 기술자(Local feature descriptors)를 코드북(Codebook)으로 양자화하고, 각 센트로이드(Centroid)에 대한 1차 및 2차 잔차 통계량(Residual statistics)을 집계하여 고차원 벡터를 생성하는 방식이다. 최근에는 Residual Enhanced Visual Vector나 RCFC와 같은 개선된 변형들이 제안되었다.
- **CNN-based Descriptors**: AlexNet 이후 CNN이 분류 작업에서 성공하며 이를 검색에 활용하려는 시도가 있었다. 특히 Babenko 등은 사전 학습된 CNN을 도메인 특정 데이터로 미세 조정(Fine-tuning)하여 성능을 높였으며, CNN 표현력이 FV보다 더 효율적으로 압축될 수 있음을 보였다.

**기존 연구의 한계 및 차별점**: 기존 CNN 기반 검색 연구들은 주로 네트워크를 블랙박스로 취급하여 결과를 보고했을 뿐, 네트워크 아키텍처나 학습 데이터가 검색 성능에 구체적으로 어떤 영향을 미치는지에 대한 체계적인 분석이 부족하였다. 또한, 관심 지점 기반의 FV와 달리 CNN은 회전 및 크기 불변성(Invariance)을 보장하는 내장 메커니즘이 없다는 점을 간과하였다. 본 논문은 이러한 세부 사항들을 실험적으로 분석하여 실무적인 가이드를 제공한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. Fisher Vectors (FV) 파이프라인
FV는 다음과 같은 절차로 생성된다.
- **특징 추출**: SIFT 기술자를 사용하며, 세 가지 샘플링 전략(DoG 기반 관심 지점, 단일 스케일 Dense, 다중 스케일 Dense)을 적용한다.
- **차원 축소 및 모델링**: PCA를 통해 SIFT의 128차원을 64차원으로 축소하고, 256개의 센트로이드를 가진 Gaussian Mixture Model(GMM)을 학습시킨다.
- **통계량 집계**: 각 센트로이드에 대해 평균과 분산에 대한 그라디언트(1차 및 2차 통계량)를 계산하여 결합한다. 결과적으로 $64 \times 256 \times 2 = 32,768$ 차원의 벡터가 생성된다.
- **정규화**: Power law normalization ($\alpha=0.5$) 후 $L_2$ 정규화를 적용한다.

### 2. CNN 기술자 추출
본 연구에서는 OxfordNet, AlexNet, PlacesNet, HybridNet 등 4가지 모델을 사용한다.
- **추출 레이어**: 각 네트워크의 마지막 4개 레이어($\text{pool5}, \text{fc6}, \text{fc7}, \text{fc8}$)에서 활성화 값을 추출하며, ReLU 적용 후 $L_2$ 정규화를 수행한다.
- **입력 처리**: 이미지의 종횡비를 유지하며 중심을 크롭하는 'Center crop' 전략을 기본으로 사용한다.

### 3. 기술자 결합 (Fusion)
CNN과 FV의 장점을 모두 취하기 위해 단순한 Early Fusion 방식을 사용한다. 두 벡터를 가중치 $\alpha$와 $1-\alpha$로 조절하여 결합(Concatenation)한다.
$$\text{Combined Descriptor} = [\alpha \cdot \text{FV}, (1-\alpha) \cdot \text{CNN}]$$
이는 거리 측정 시 각 기술자의 가중 합산 거리와 동일한 효과를 준다.

### 4. 변환 불변성 확보를 위한 Database Pooling
CNN의 회전 및 크기 취약성을 해결하기 위해 데이터베이스 이미지들에 대해 다음과 같은 풀링 기법을 제안한다.
- **Max-pooling**: 데이터베이스 이미지를 일정 각도($\pm P^\circ$)나 스케일로 변형시켜 여러 개의 기술자를 추출한 후, 각 차원별 최댓값을 취한다.
- **Min-dist (PWL)**: 모든 변형된 기술자를 저장하는 대신, 회전/스케일에 따른 기술자의 궤적(Manifold)을 조각별 선형 근사(Piece-wise Linear approximation)로 모델링하고, 쿼리 벡터와의 최단 거리를 계산한다.

## 📊 Results

### 1. CNN 및 FV 최적 설정 실험
- **CNN 크롭 전략**: Center crop과 Squish(이미지를 강제로 정방형으로 리사이즈)가 Padding보다 우수한 성능을 보였다.
- **CNN 레이어**: 대부분의 데이터셋에서 $\text{fc6}$ 레이어가 가장 높은 성능을 보였다. 이는 최종 레이어($\text{fc8}$)는 너무 추상적인 세만틱 정보만 가지고 있고, 앞쪽 레이어는 너무 저수준 정보만 가지고 있기 때문이다.
- **네트워크 깊이 및 데이터**: 학습 데이터와 테스트 데이터의 도메인이 유사할 때만 더 깊은 네트워크(OxfordNet)가 AlexNet보다 우수하였다. 또한, scene-centric 데이터(Holidays, Oxbuild)에서는 PlacesNet이, object-centric 데이터(UKBench)에서는 AlexNet이 더 효과적이었다.
- **FV 샘플링**: 텍스처가 풍부한 장면(Holidays)에서는 Dense 샘플링이 유리하지만, 객체의 회전과 크기 변화가 심한 데이터(Graphics)에서는 DoG 기반의 Sparse 샘플링이 훨씬 강력했다.

### 2. 정량적 비교 및 결합 성능
- **단일 기술자 비교**: 데이터셋의 특성에 따라 우위가 달라진다. 일반적인 벤치마크에서 CNN이 FV보다 높은 성능을 보이는 경우가 많았으나, 특정 조건에서는 FV가 여전히 강력했다.
- **Fusion 결과**: CNN과 FV를 결합했을 때 모든 데이터셋에서 성능 향상이 관찰되었다. 특히 $\alpha=0.3 \sim 0.4$ 범위에서 피크 성능이 나타났으며, 이는 FV가 단순 보조 수단이 아니라 상당한 기여를 하고 있음을 시사한다.

### 3. 변환 강건성 실험
- **회전(Rotation)**: CNN은 $10^\circ$ 이상의 회전에서 성능이 급격히 하락한다. 반면 FV(DoG)는 매우 강건하다. 제안된 Max-pooling과 Min-dist (PWL) 기법을 적용하면 CNN의 회전 강건성을 획기적으로 높일 수 있었다.
- **크기(Scale)**: CNN은 크기 변화에 매우 강건하여 0.25배까지는 성능 하락이 적었다. 반면 FV는 크기가 작아질수록 성능이 더 빠르게 하락하였다.

## 🧠 Insights & Discussion

본 논문은 CNN과 FV라는 두 가지 서로 다른 패러다임의 기술자가 각각 가진 강점과 약점을 명확히 분석하였다.

**강점 및 발견**:
- CNN은 전반적인 변별력(Discriminativeness)이 높고 특히 크기 변화에 강하지만, 회전에는 매우 취약하다.
- FV(DoG)는 회전 불변성이 내장되어 있어 기하학적 변형이 심한 환경에서 유리하다.
- **"변별력-불변성 트레이드오프"**: Dense 샘플링은 불변성을 희생하는 대신 변별력을 높여 성능을 올리지만, 변형이 심한 데이터에서는 오히려 독이 된다.

**한계 및 비판적 해석**:
- 본 연구에서 제안한 Database Pooling 기법은 사후 처리 방식(After-thought)으로, 데이터베이스의 저장 공간과 연산 비용을 증가시킨다. 학습 단계에서부터 회전/크기 불변성을 학습하는 아키텍처를 설계하는 것이 근본적인 해결책이 될 것이다.
- 단순한 Early Fusion이 효과적임을 보였으나, 더 정교한 랭킹 알고리즘이나 후처리(Reranking) 기법을 적용했을 때의 결과는 다루지 않았다.

## 📌 TL;DR

본 논문은 이미지 검색을 위한 CNN과 Fisher Vector(FV) 기술자를 체계적으로 비교 분석한 가이드라인이다. **CNN은 전반적인 변별력과 크기 불변성이 우수**하고, **FV(DoG)는 회전 불변성이 탁월**하다는 것을 실험적으로 입증하였다. 결론적으로 어느 하나가 절대적으로 우월하기보다는 **두 기술자를 결합(Fusion)하여 사용하는 것이 최적의 성능**을 내며, CNN의 회전 취약성은 데이터베이스 측의 Max-pooling이나 Min-dist (PWL) 기법으로 보완할 수 있음을 제시하였다. 이 연구는 향후 특정 도메인의 이미지 검색 시스템을 구축할 때 기술자 선택과 최적화 전략을 수립하는 데 중요한 참조 자료가 될 것이다.