# Unsupervised Deep Metric Learning via Auxiliary Rotation Loss

Xuefei Cao, Bor-Chun Chen, Ser-Nam Lim (2019)

## 🧩 Problem to Solve

본 논문은 Deep Metric Learning(DML)에서 발생하는 레이블 데이터 의존성 문제를 해결하고자 한다. Metric Learning의 핵심은 유사한 샘플은 가깝게, 서로 다른 클래스의 샘플은 멀게 배치하는 임베딩 공간을 학습하는 것이나, 이를 위해서는 대규모의 정교하게 레이블링된 데이터셋이 필수적이다. 특히 세밀한 분류가 필요한 fine-grained 데이터셋의 경우, 도메인 전문가의 주석이 필요하여 레이블 획득 비용이 매우 높다는 문제가 있다.

따라서 본 연구의 목표는 레이블 없이도 유의미한 임베딩 공간을 학습할 수 있는 Unsupervised Deep Metric Learning(UDML) 프레임워크를 구축하는 것이며, 특히 클러스터링 기반의 의사 레이블(pseudo-labels) 생성 과정에서 발생하는 불안정성을 해결하는 데 집중한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 클러스터링을 통한 Metric Learning과 자기지도학습(Self-Supervised Learning, SSL)의 pretext task를 결합하여 학습의 안정성을 높이는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **UDML-SS 프레임워크 제안**: k-means 클러스터링을 통해 얻은 의사 레이블을 사용하여 Metric Learning을 수행하고, 이를 이미지 회전 예측(Rotation Prediction)이라는 자기지도학습 과제와 결합한 다중 작업 학습(multi-task learning) 구조를 제안한다.
2. **학습 안정화**: 클러스터링 결과가 항상 정확하지 않다는 점(unreliable assignments)에 주목하여, 보조 손실 함수인 Rotation Loss를 도입함으로써 모델이 클러스터링 결과에만 과도하게 의존하지 않고 데이터 자체의 유의미한 특징(representation)을 학습하도록 유도한다.
3. **성능 입증**: CUB-200-2011, Cars-196, Stanford Online Products 등 주요 벤치마크 데이터셋에서 기존의 최신 비지도 학습 방법론들을 크게 상회하는 성능을 보였으며, 일부 데이터셋에서는 지도 학습 방법론에 근접하는 결과를 달성하였다.

## 📎 Related Works

**1. Metric Learning**
기존의 지도 학습 기반 Metric Learning은 Contrastive Loss, Triplet Loss 등을 통해 쌍(pair)이나 삼조(triplet) 간의 거리를 조절한다. 최근에는 Multi-similarity Loss와 같이 배치 내의 모든 샘플 간 관계를 활용하거나, Semi-hard mining과 같은 기법을 통해 학습 효율을 높이는 방향으로 발전하였다. 그러나 이 모든 방법은 정답 레이블이 있다는 가정하에 작동한다.

**2. Self-supervised Representation Learning**
레이블 없이 데이터 자체에서 정답을 찾는 방법으로, 이미지 패치의 상대적 위치 예측, 컬러라이제이션(colorization), 이미지 회전 각도 예측 등이 제안되었다. 특히 Gidaris 등이 제안한 이미지 회전 예측은 단순하면서도 강력한 feature representation을 학습할 수 있음이 입증되었다.

**3. Deep Clustering 및 Unsupervised Metric Learning**
DeepCluster와 같은 연구는 k-means 클러스터링과 분류(classification) 작업을 반복하여 특징을 학습한다. 비지도 Metric Learning 분야에서는 매니폴드 거리(manifold distance)를 이용한 하드 샘플 마이닝이나, 데이터 증강을 통한 인스턴스 기반의 변별력 학습 등이 시도되었으나, 본 논문은 이를 Metric Learning Loss와 SSL task의 결합으로 확장하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

UDML-SS는 특징 추출기(Feature Extractor) 하나에 두 개의 헤드(Metric Learning Head, Rotation Prediction Head)가 붙은 구조를 가진다. 전체 프로세스는 **[임베딩 추출 $\rightarrow$ k-means 클러스터링 $\rightarrow$ 의사 레이블 생성 $\rightarrow$ 손실 함수 최적화 $\rightarrow$ 임베딩 업데이트]** 순으로 반복적으로 수행된다.

### 주요 구성 요소 및 학습 절차

**1. 특징 추출 및 임베딩**
입력 이미지 $x$는 CNN 기반의 특징 추출기 $f_{\theta_1}$를 통해 특징 벡터 $w$로 변환되고, 다시 $\theta_2$ 파라미터를 가진 매핑 함수 $g_{\theta_2}$를 통해 단위 길이로 정규화된 임베딩 벡터 $z$로 투영된다. 두 벡터 간의 유사도는 코사인 유사도로 정의된다:
$$S(x_i, x_j) = \langle g_{\theta_2}(f_{\theta_1}(x_i)), g_{\theta_2}(f_{\theta_1}(x_j)) \rangle$$

**2. Multi-similarity Loss ($L_{MS}$)**
본 논문은 최신 성능이 검증된 Multi-similarity loss를 채택하였다. 클러스터링으로 얻은 의사 레이블 $y$를 기준으로 하드 샘플을 마이닝한다.

- **Negative Pair**: $S(x_i, x_j) > \min_{y_h=y_i} S(x_i, x_h) - \epsilon$ 인 경우
- **Positive Pair**: $S(x_i, x_k) < \max_{y_h \neq y_i} S(x_i, x_h) + \epsilon$ 인 경우
위 조건으로 선택된 쌍들을 이용하여 다음의 손실 함수를 최소화한다:
$$L_{MS} = \sum_{i=1}^{n} \left( \frac{1}{\alpha} \log(1 + \sum_{l \in P_i} e^{-\alpha(S_{il}-\lambda)}) + \frac{1}{\beta} \log(1 + \sum_{l \in N_i} e^{\beta(S_{il}-\lambda)}) \right)$$

**3. Rotation Prediction Loss ($L_{rot}$)**
모델의 안정성을 위해 이미지를 $0^\circ, 90^\circ, 180^\circ, 270^\circ$로 회전시킨 뒤, 회전 각도를 맞추는 4클래스 분류 문제를 푼다. 특징 추출기 $f_{\theta_1}$ 뒤에 붙은 회전 예측 헤드 $h_{\theta_3}$를 통해 계산되며, 교차 엔트로피 손실(Cross-Entropy Loss)을 사용한다:
$$L_{rot} = \frac{1}{n} \sum_{k=1}^{4} \sum_{i=1}^{n} L(h_{\theta_3}(f_{\theta_1}(x_{ik})), z_{ik})$$

**4. 통합 손실 함수**
최종 목적 함수는 Metric Learning 손실과 Rotation 손실의 가중 합으로 정의된다:
$$L_{UDML-SS} = L_{MS}(\theta_1, \theta_2, x, y) + \eta L_{rot}(\theta_1, \theta_3, x, z)$$
여기서 $\eta$는 두 작업 간의 균형을 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CUB-200-2011, Cars-196, Stanford Online Products (Product)
- **백본**: ImageNet으로 사전 학습된 Inception-V1 (일부 실험에서 Inception-V2, ResNet 사용)
- **평가 지표**: Recall@K (이미지 검색 성능), NMI (클러스터링 성능)

### 정량적 결과

UDML-SS는 모든 데이터셋에서 기존의 비지도 학습 방법론(Instance, DeepCluster, MOM 등)을 크게 앞질렀다.

- **CUB-200-2011**: Recall@1 기준 Instance 방법론 대비 **+8.5%** 상승하여 54.7% 달성.
- **Cars-196**: Recall@1 기준 **+3.8%** 상승하여 45.1% 달성.
- **Product**: Recall@1 기준 **+14.6%** 상승하여 63.5% 달성 ($k=10,000$ 기준).

### 주요 분석 및 어블레이션 연구

- **Rotation Loss의 효과**: Rotation Loss를 제거($\eta=0$)했을 때, 특히 Cars 데이터셋에서 Recall@1 성능이 약 7% 하락하였다. 이는 클러스터링이 불안정한 데이터셋일수록 SSL task가 특징 학습의 가이드라인 역할을 하여 안정성을 부여함을 시사한다.
- **임베딩 차원**: 임베딩 차원이 64에서 512까지 증가함에 따라 성능이 지속적으로 향상되는 경향을 보였다.
- **초기값 영향**: 사전 학습된 네트워크 없이 Random Initialization 상태에서 시작하더라도, UDML-SS는 다른 비지도 학습 방법론들보다 월등히 높은 성능을 보였다(Product 데이터셋 기준).

## 🧠 Insights & Discussion

**강점 및 성과**
본 논문은 비지도 학습의 고질적인 문제인 '불안정한 의사 레이블' 문제를 '보조적인 자기지도학습 과제'로 해결하였다. 특히 Multi-similarity loss라는 강력한 지도 학습 도구를 비지도 환경으로 성공적으로 이식하였으며, 이를 통해 비지도 학습임에도 불구하고 일부 지도 학습 모델에 근접하는 성능을 낸 점이 고무적이다.

**한계 및 비판적 해석**
실험 결과에서 Cars 데이터셋의 성능이 타 데이터셋에 비해 현저히 낮게 나타났다. 저자들은 이를 자동차 로고와 같은 매우 미세한(fine-grained) 차이를 비지도 학습만으로는 구분하기 어렵기 때문이라고 분석하였다. 이는 UDML-SS가 전반적인 형태나 색상 등의 특징은 잘 잡아내지만, 사람이 식별하는 결정적인 미세 특징(discriminative fine-grained feature)을 스스로 학습하는 데에는 여전히 한계가 있음을 보여준다.

또한, 사전 학습된 Inception-V1 모델의 강력한 초기 신호가 성능에 큰 영향을 미쳤을 가능성이 크다. 비록 Random Initialization 실험을 수행하였으나, 사전 학습 모델을 사용했을 때의 성능 향상 폭이 매우 커서 실제 적용 시에는 강력한 Pre-trained backbone에 의존하게 될 가능성이 높다.

## 📌 TL;DR

본 연구는 **클러스터링 기반의 의사 레이블 생성**과 **이미지 회전 예측(Rotation Prediction)**이라는 자기지도학습 태스크를 결합한 비지도 Metric Learning 프레임워크(UDML-SS)를 제안한다. 이를 통해 클러스터링의 불안정성을 극복하고 안정적으로 유의미한 임베딩 공간을 학습할 수 있게 되었으며, CUB, Cars, Product 데이터셋에서 기존 SOTA 비지도 방법론들을 압도하는 성능을 기록하였다. 이 연구는 레이블링 비용이 높은 도메인에서 고성능의 이미지 검색 및 식별 시스템을 구축하는 데 중요한 기초가 될 수 있다.
