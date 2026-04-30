# Associative Adversarial Networks

Tarik Arici, Asli Celikyilmaz (2016)

## 🧩 Problem to Solve

본 논문은 Generative Adversarial Network (GAN)의 학습 과정에서 발생하는 불안정성과 학습의 어려움을 해결하고자 한다. 일반적인 GAN 프레임워크에서 생성자($G$)는 균등 분포(uniform distribution)를 따르는 화이트 노이즈 $z$를 입력받아 데이터 공간으로 매핑하는 작업을 수행한다. 하지만 저자들은 이처럼 아무런 구조가 없는 평탄한(flat) 표현 공간에서 복잡한 데이터 공간으로의 매핑을 학습하는 것이 $G$에게 매우 어려운 과제이며, 이것이 학습 중 생성자가 붕괴(collapse)하거나 수렴에 실패하는 주요 원인 중 하나라고 주장한다.

따라서 본 연구의 목표는 $G$의 입력으로 단순한 노이즈 대신, 데이터의 고수준 특징(high-level features)을 담고 있는 **연상 메모리(associative memory)**를 도입하여 $G$의 학습 부담을 완화하고 전체 시스템의 학습 안정성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 생성자와 판별자 사이에 확률적 생성 모델인 **Restricted Boltzmann Machine (RBM)**을 배치하여 고수준 연상 메모리로 사용하는 것이다.

판별자($D$)가 데이터로부터 추출한 중간 계층의 특징 표현 공간(representation space)에 대한 확률 분포를 RBM이 학습하게 하고, $G$는 이 RBM에서 샘플링된 값을 입력으로 받는다. 즉, $G$는 완전히 무작위인 노이즈가 아니라, $D$가 인식한 데이터의 잠재적 특징들의 분포 내에서 샘플링된 값을 입력받아 데이터를 생성하게 된다. 이는 $G$가 학습해야 할 매핑의 난이도를 낮추어 주는 효과를 제공한다.

## 📎 Related Works

저자들은 GAN의 학습 안정성을 높이기 위한 기존의 여러 기법(minibatch discrimination, historical averaging 등)과 DCGAN의 아키텍처 제안들을 언급한다. 

특히 본 연구의 접근 방식은 **Deep Belief Networks (DBN)**의 학습 알고리즘 및 **wake-sleep algorithm**과 유사한 점이 있다. DBN은 최상위 계층에 무방향 연상 메모리를 두고 하위 계층에 방향성 생성 모델을 둔다. 본 논문의 AAN 구조에서도 판별자($D$)를 통한 상향 패스(up-pass)와 생성자($G$)를 통한 하향 패스(down-pass)가 RBM을 중심으로 연결되어 있다는 점에서 이러한 구조적 유사성을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조
AAN은 세 가지 네트워크의 결합으로 구성된다:
1. **Discriminator ($D$):** 데이터를 입력받아 특징을 추출하고 진위 여부를 판별한다.
2. **Associative Memory (RBM):** $D$의 중간 계층 활성화 값들의 분포를 학습하고 샘플을 생성한다.
3. **Generator ($G$):** RBM에서 생성된 샘플을 입력받아 데이터 공간으로 매핑한다.

판별자 $D$의 연산을 $D(x) = C(F(x))$라고 정의할 때, $F(x)$는 중간 계층의 특징 추출기이며 $C(y)$는 이후의 분류 단계이다. RBM은 이 $f = F(x)$의 분포 $p_f$를 학습한다.

### RBM 및 학습 절차
RBM은 가시층(visible layer) $v$와 은닉층(hidden layer) $h$로 구성된 에너지 기반 모델이다. 에너지 함수 $E(v, h)$는 다음과 같이 정의된다:

$$E(v, h) = \frac{1}{2} \sum_{i} v_i^2 - \sum_{i, j} v_i h_j w_{ij} - \sum_{i} v_i b_i - \sum_{j} h_j c_j$$

이에 따른 확률 분포 $P(v, h)$와 분배 함수(partition function) $Z$는 다음과 같다:

$$P(v, h) = \frac{e^{-E(v, h)}}{Z}, \quad Z = \sum_{x, y} e^{-E(x, y)}$$

RBM의 학습에는 **Contrastive Divergence (CD)** 알고리즘이 사용되며, 가시층에 데이터를 클램핑(clamping)한 후 Gibbs sampling을 통해 음의 단계(negative phase) 샘플을 생성하여 가중치를 업데이트한다.

### AAN의 목적 함수
AAN의 전체 최적화 문제는 다음과 같은 수식으로 표현된다:

$$\min_{G} \max_{\hat{p}_f, D} = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{f \sim \hat{p}_f(f)}[\log(1 - D(G(f)))] + \mathbb{E}_{f \sim p_f(f)}[\log \hat{p}_f]$$

여기서 첫 번째와 두 번째 항은 일반적인 GAN의 적대적 손실 함수이며, 세 번째 항은 연상 메모리($\hat{p}_f$)가 판별자의 특징 공간 분포($p_f$)를 정확하게 추정하도록 강제하는 최대 우도(maximum likelihood) 항이다.

### 상세 구현 사항
- **활성화 함수:** $D$에는 LeakyReLU를 사용하였으나, RBM의 가시층과 연결되는 지점에는 binary RBM과의 호환성을 위해 $\tanh$ 활성화 함수를 사용하였다. $G$에는 ReLU를 사용하였다.
- **학습 알고리즘:** GAN 부분은 Adam optimizer를, RBM 부분은 momentum이 적용된 SGD를 사용하였다.

## 📊 Results

### 실험 설정
- **데이터셋:** CelebA(얼굴 이미지) 및 MNIST 데이터셋을 사용하였다.
- **측정 및 분석:** $D$와 $G$의 수렴도 분석, RBM의 차원($100$ vs $1000$)에 따른 생성 이미지의 변화를 관찰하였다.

### 주요 결과
1. **RBM 차원에 따른 영향:** 
   - $1000 \times 1000$ RBM을 사용했을 때, Gibbs sampling 단계가 진행됨에 따라 얼굴의 표정이나 특징이 서서히 변한다. 이는 샘플이 분포의 다른 모드(mode)로 이동하는 데 시간이 오래 걸림을 의미한다.
   - $100 \times 100$ RBM을 사용했을 때는 단 한 번의 샘플링만으로도 성별이나 인종 등 큰 특징이 급격히 변한다. 이는 낮은 차원의 공간에서 특징들이 더 균일하게 분포되어 있어, Gibbs sample이 서로 다른 모드 사이를 더 빠르게 점프할 수 있음을 보여준다.

2. **수렴 분석:**
   - 일반적인 GAN(화이트 노이즈 입력)에 비해 AAN은 $G$가 $D$의 학습 속도를 더 잘 따라가는 경향을 보인다. 
   - 수치적으로 $\mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)]$와 $\mathbb{E}_{f \sim \hat{p}_f(f)}[\log D(G(f))]$의 비율을 분석했을 때, AAN이 $G$의 학습 과제를 완화시켜 $D$에 의해 더 효과적으로 가이드될 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 $G$의 입력값을 단순한 노이즈에서 구조화된 특징 공간의 샘플로 변경함으로써, 적대적 학습의 고질적인 문제인 $G$와 $D$ 사이의 학습 속도 불균형 문제를 완화하였다. 특히 판별자가 찾은 고수준 표현 공간을 연상 메모리로 활용함으로써, 생성자가 데이터의 매니폴드(manifold) 상의 유의미한 지점에서 시작할 수 있도록 돕는다는 점이 핵심적인 통찰이다.

### 한계 및 논의사항
- **메모리 붕괴 가능성:** 저자들은 수식 (1)을 통해 연상 메모리 자체가 퇴화된 확률 분포로 붕괴(collapse)될 가능성이 있음을 명시하였다. 이는 GAN의 mode collapse와 유사한 문제이다.
- **이론적 근거 부족:** 실험적으로는 효능을 입증하였으나, 보다 엄밀한 이론적 분석이 필요함을 인정하고 있다. 이를 해결하기 위해 향후 연구로 엔트로피 최대화 정규화(entropy-maximizing regularizers) 도입을 제시하였다.

## 📌 TL;DR

본 논문은 GAN의 생성자($G$)가 단순 노이즈를 데이터로 매핑하는 어려움을 해결하기 위해, 판별자($D$)의 중간 특징 공간을 학습하는 **RBM 기반의 연상 메모리**를 도입한 **AAN(Associative Adversarial Networks)**을 제안한다. RBM이 학습한 고수준 특징 분포에서 샘플링된 값을 $G$의 입력으로 사용함으로써 학습의 안정성을 높이고 수렴 속도를 개선하였다. 이 연구는 생성 모델의 입력 공간을 구조화하는 것이 적대적 학습의 난이도를 낮추는 효과적인 방법이 될 수 있음을 시사한다.