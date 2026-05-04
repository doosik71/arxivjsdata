# HIDDEN MARKOV RANDOM FIELDS AND CUCKOO SEARCH METHOD FOR MEDICAL IMAGE SEGMENTATION

EL-Hachemi Guerrout, Ramdane Mahiou, Dominique Michelucci, Boukabene Randa, and Ouali Assia (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상 진단 과정의 핵심 단계인 의료 영상 분할(Medical Image Segmentation)의 자동화 및 강건성 확보 문제를 해결하고자 한다. 특히 뇌 MRI(Magnetic Resonance Imaging) 영상의 경우, 영상 획득 장치의 무선 주파수 코일이나 조명의 부분적 변화와 같은 다양한 요인으로 인해 노이즈와 불균일성 아티팩트(Non-uniformity artifact)가 발생하며, 이로 인해 정확한 분할이 매우 어렵다.

본 연구의 구체적인 목표는 뇌 영상을 회백질(Gray Matter, GM), 백질(White Matter, WM), 그리고 뇌척수액(Cerebrospinal Fluid, CSF)의 세 가지 클래스로 정확하게 분할하는 자동화된 알고리즘을 제안하고 그 성능을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Hidden Markov Random Fields (HMRF) 모델을 통해 영상 분할 문제를 에너지 함수 최소화 문제로 정의하고, 이를 해결하기 위해 자연 모사 메타 휴리스틱 알고리즘인 Cuckoo Search (CS)를 최적화 도구로 사용하는 것이다. 

전통적인 최적화 방법은 MAP(Maximum A Posteriori) 기준의 목적 함수가 비선형적이고 복잡하며 여러 지역 최적점(Local maxima)을 가질 수 있다는 한계가 있다. 이를 극복하기 위해 Lévy flights를 이용한 탐색 능력이 뛰어난 Cuckoo Search 알고리즘을 도입하여 전역 최적해를 효율적으로 찾고자 하였다. 특히, 표준 CS뿐만 아니라 개선된 CS(Improved CS)와 자기 적응형 수정 CS(Auto adaptive modified CS)의 세 가지 변형 알고리즘을 비교 분석하여 최적의 조합을 제시하였다.

## 📎 Related Works

논문에서는 영상 분할을 위해 HMRF 모델과 메타 휴리스틱 알고리즘의 필요성을 언급한다. HMRF는 영상의 공간적 상관관계를 모델링하여 보다 매끄러운 분할 결과를 얻을 수 있는 강력한 모델로 알려져 있다. 

최적화 알고리즘 측면에서는 2009년 Xin-She Yang과 Suash Deb가 제안한 Cuckoo Search (CS)를 소개한다. CS는 일부 뻐꾸기 종의 탁란(Brood parasitism) 습성에서 영감을 얻은 알고리즘으로, 단순한 등방성 무작위 보행(Isotropic random walks) 대신 Lévy flights를 사용하여 탐색 효율을 높인 것이 특징이다. 기존의 최적화 방식들이 지역 최적점에 빠지기 쉬운 반면, CS는 보다 광범위한 탐색 영역을 가짐으로써 복잡한 HMRF 에너지 함수 최소화 문제에 적합하다는 점을 차별점으로 제시한다.

## 🛠️ Methodology

### 1. HMRF 모델 및 에너지 함수
영상을 $M$개의 사이트로 구성된 $y = (y_1, y_2, ..., y_M)$로 정의하고, 분할된 영상을 $x = (x_1, x_2, ..., x_M)$로 정의한다. 여기서 $y_s$는 픽셀 값($0 \sim 255$)이며, $x_s$는 클래스($1 \sim K$)를 나타낸다.

분할 문제는 에너지 함수 $\Psi(\mu)$를 최소화하는 평균값 $\mu^* = (\mu_1, ..., \mu_K)$를 찾는 문제로 귀결된다.
$$\mu^* = \arg \min_{\mu \in \Omega_\mu} \{ \Psi(\mu) \}$$
에너지 함수 $\Psi(\mu)$는 다음과 같이 정의된다:
$$\Psi(\mu) = \sum_{j=1}^{K} f(\mu_j)$$
$$f(\mu_j) = \sum_{s \in S_j} \left[ \ln(\sigma_j) + \frac{(y_s - \mu_j)^2}{2\sigma_j^2} \right] + \frac{B}{T} \sum_{\{s,t\}} (1 - 2\delta(x_s, x_t))$$

여기서 $\sigma_j$는 클래스 $j$의 표준편차, $B$는 균질한 영역의 크기를 조절하는 상수, $T$는 온도 파라미터, $\delta$는 Kronecker’s delta이다. 픽셀 $y_s$는 가장 가까운 평균값 $\mu_j$를 가진 클래스로 할당된다.

또한, 최적화 기법을 무제한 영역으로 확장하기 위해 $\mu \in \mathbb{R}^K$ 범위에서 작동하도록 함수 $F(\mu_j)$를 재정의하여 $\mu_j$가 $[0, 255]$ 범위를 벗어날 경우 패널티를 부여하는 방식을 사용한다.

### 2. Cuckoo Search (CS) 최적화 절차
CS 알고리즘은 각 둥지(Nest)를 하나의 잠재적 해 $\mu_{i,t}$로 간주하며, 다음과 같은 절차를 따른다:

1. **새로운 뻐꾸기 생성 (Lévy flight):**
   $$c_{i,t} := \mu_{i,t} + \alpha \times \text{step} \otimes (\mu_{i,t} - \text{best}_t) \otimes \text{randn}(1, K)$$
   여기서 $\alpha$는 단계 크기 조절 인자이며, $\text{best}_t$는 현재까지 찾은 최적해이다.

2. **둥지 업데이트:** 새로운 해 $c_{i,t}$의 에너지 값이 기존 둥지의 에너지 값보다 낮을 경우 이를 교체한다.
   $$\mu_{i,t} := \begin{cases} c_{i,t} & \text{if } \Psi(c_{i,t}) \le \Psi(\mu_{i,t}) \\ \mu_{i,t} & \text{otherwise} \end{cases}$$

3. **나쁜 둥지 제거 및 재구성:** 확률 $p_a$로 성능이 낮은 둥지를 버리고, 선택적 무작위 보행(Selective random walks)을 통해 새로운 해를 생성한다.
   $$\mu_{i,t+1} := \mu_{i,t} + [\text{rand}()] \otimes [H(p_a - \text{rand}())] \otimes (\mu_{a,t} - \mu_{b,t})$$
   여기서 $H$는 Heaviside 함수이며, $\mu_{a,t}$와 $\mu_{b,t}$는 무작위로 선택된 두 해이다.

## 📊 Results

### 실험 설정
- **데이터셋:** BrainWeb (시뮬레이션 MRI) 및 IBSR (전문가 수동 분할 MRI) 데이터베이스.
- **평가 지표:** Dice Coefficient (DC).
  $$DC = \frac{2|\hat{A} \cap A^*|}{|\hat{A} \cup A^*|}$$
  ($\hat{A}$는 예측 결과, $A^*$는 Ground Truth)
- **파라미터:** 둥지 수 $n=30$, 온도 $T=4$, 최대 세대 수 $\text{MaxGeneration}=100$.

### 정량적 결과
1. **실행 시간:**
   - BrainWeb의 경우 Standard 및 Improved CS가 $85 \sim 105$초로 가장 빨랐으며, Auto-adaptive CS는 $130 \sim 140$초가 소요되었다.
   - IBSR의 경우 Standard 및 Improved CS가 $125 \sim 150$초, Auto-adaptive CS는 $230 \sim 250$초가 소요되었다.

2. **Dice Coefficient (DC):**
   - **BrainWeb:** HMRF-CS_Improved가 평균적으로 가장 높은 DC 값을 보였으며, 특히 테스트 3에서 평균 0.974의 높은 정확도를 기록하였다.
   - **IBSR:** 전반적으로 BrainWeb보다 낮은 성능을 보였으나, GM과 WM에서는 0.9 수준의 높은 값을 보였다. 다만 CSF 영역의 분할 성능은 0.54~0.55 수준으로 상대적으로 낮게 나타났다.

### 정성적 결과
시각적 결과 분석에서 HMRF-CS_Improved 방법이 Ground Truth와 매우 유사한 분할 결과를 생성함을 확인하였다. 특히 뇌의 주요 조직인 GM, WM, CSF의 경계가 비교적 명확하게 구분되었다.

## 🧠 Insights & Discussion

본 연구는 HMRF 모델의 에너지 최소화 문제를 해결하기 위해 Cuckoo Search 알고리즘을 성공적으로 결합하였다. 실험 결과, 특히 **HMRF-CS_Improved** 조합이 다른 변형 알고리즘보다 성능과 시간 효율성 측면에서 우위에 있음을 확인하였다. CS 알고리즘이 단순하면서도 강건하며, MRI 영상과 같이 복잡한 최적화가 필요한 문제에서 효과적인 도구가 될 수 있음을 시사한다.

다만, 다음과 같은 한계점이 존재한다:
- **CSF 분할 성능 저하:** IBSR 데이터셋에서 CSF의 DC 값이 낮게 나타났는데, 이는 CSF의 강도(intensity) 분포가 다른 조직과 겹치거나 영역이 좁아 분할이 어렵기 때문으로 추정된다.
- **비교 대상의 부족:** 최신 딥러닝 기반의 세그멘테이션 방법론(예: U-Net 등)과의 비교 분석이 이루어지지 않아, 현대적인 기준에서의 경쟁력을 판단하기 어렵다.
- **파라미터 분석 미흡:** $n, T, \alpha, p_a$ 등 주요 파라미터가 결과에 미치는 영향에 대한 통계적 분석이 충분하지 않다.

## 📌 TL;DR

본 논문은 MRI 뇌 영상 분할을 위해 **Hidden Markov Random Fields(HMRF)** 모델과 **Cuckoo Search(CS)** 최적화 알고리즘을 결합한 방법론을 제안하였다. 세 가지 CS 변형 알고리즘 중 **Improved CS**가 가장 우수한 성능을 보였으며, 특히 시뮬레이션 데이터(BrainWeb)에서 매우 높은 정확도를 기록하였다. 이 연구는 메타 휴리스틱 알고리즘을 의료 영상 분할의 에너지 최소화 문제에 적용하여 강건한 자동 분할 가능성을 제시했다는 점에서 의의가 있다.