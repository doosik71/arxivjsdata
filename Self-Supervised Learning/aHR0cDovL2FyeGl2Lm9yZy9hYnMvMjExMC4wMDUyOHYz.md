# Do Self-Supervised and Supervised Methods Learn Similar Visual Representations?

Tom George Grigg, Dan Busbridge, Jason Ramapuram, Russ Webb (2021)

## 🧩 Problem to Solve

본 논문은 시각적 자기지도학습(Self-Supervised Learning, SSL)과 지도학습(Supervised Learning, SL)이 최종적으로 학습하는 표현(representation)이 서로 유사한지에 대한 문제를 해결하고자 한다. 최근 다양한 SSL 기법들이 지도학습의 성능에 근접하는 성과를 보이고 있음에도 불구하고, 실제로 이들이 신경망 내부에서 어떤 표현을 학습하는지에 대한 심층적인 분석은 부족한 상황이다.

이 문제의 중요성은 SSL이 클래스 레이블 없이 어떻게 경쟁력 있는 표현을 구축하는지를 이해함으로써, 향후 더 효율적인 보조 작업(auxiliary task)을 설계하고 SSL의 작동 원리를 이론적으로 규명하는 데 있다. 따라서 본 논문의 목표는 동일한 아키텍처 내에서 대조 학습(Contrastive Learning) 기반의 SSL 알고리즘인 SimCLR와 지도학습(SL)이 생성하는 표현의 유사성을 비교 분석하는 것이다.

## ✨ Key Contributions

본 연구의 핵심적인 직관은 SSL과 SL이 서로 다른 학습 목적 함수를 가지더라도, 네트워크의 중간 계층에서는 유사한 특징을 학습할 수 있다는 가설에서 출발한다. 주요 기여 사항은 다음과 같다.

- **표현의 층별 유사성 발견**: SimCLR와 SL이 네트워크의 중간 계층에서는 유사한 표현을 학습하지만, 마지막 몇 개의 계층에서는 급격하게 서로 다른 방향으로 발산한다는 점을 밝혀냈다.
- **학습 목적의 반영**: 발산하는 최종 계층들을 분석한 결과, SimCLR는 데이터 증강에 대한 불변성(augmentation invariance)을 학습하는 반면, SL은 클래스 구조(class structure)에 강력하게 피팅된다는 점을 확인하였다.
- **암시적 학습 관계**: 대조 학습 목적 함수(contrastive objective)가 중간 계층에서 지도학습 목적 함수를 암시적으로 학습하고 있으나, 그 역(SL이 SSL의 불변성을 학습하는 것)은 성립하지 않음을 발견하였다.
- **성능의 원천 규명**: SimCLR의 강력한 하위 작업(downstream task) 성능은 최종 표현의 구조적 일치보다는, 중간 계층에서 학습된 클래스 정보가 풍부한 표현(class-informative representations)에서 기인한다는 통찰을 제시하였다.

## 📎 Related Works

본 논문은 시각적 SSL 중에서도 특히 **Multi-view visual SSL**에 집중하며, 그 대표 주자인 **SimCLR**를 분석 대상으로 삼는다. 기존의 SSL 연구들은 주로 하위 작업의 정확도(accuracy)나 성능 지표에 집중했으나, 본 논문은 신경망 내부의 표현 공간 자체를 직접 비교하는 분석적 접근을 취한다.

신경망 표현을 비교하기 위해 본 논문은 **Centered Kernel Alignment (CKA)**를 사용한다. CKA는 고차원 표현 공간의 정렬 문제와 분포 차이를 효과적으로 처리할 수 있는 유사도 지표로, 이전 연구(Kornblith et al., 2019)에서 그 유용성이 입증된 바 있다. 본 연구는 CKA를 아키텍처 간 비교가 아닌, 동일 아키텍처 내에서 서로 다른 학습 방법론 간의 비교에 적용함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 및 분석 구조

분석을 위해 **ResNet-50 (R50)** 아키텍처를 공통 백본으로 사용하며, 데이터셋은 **CIFAR-10**을 이용한다. 비교 대상은 SimCLR로 학습된 모델과 지도학습(SL)으로 학습된 모델이다.

### 핵심 알고리즘 및 손실 함수

**SimCLR**는 이미지 $x$에 대해 두 가지 서로 다른 증강(augmentation) 뷰 $v_t(x)$와 $v_{t'}(x)$를 생성하고, 이들 간의 유사성을 극대화하며 다른 이미지와의 유사성은 최소화하는 **InfoNCE** 손실 함수를 사용한다.

$$L_{InfoNCE}^{i,j} = -\log \frac{\exp(\text{sim}(v_i, v_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(v_i, v_k)/\tau)}$$

여기서 $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$는 코사인 유사도이며, $\tau$는 온도 매개변수이다.

### 표현 비교 방법 (CKA)

두 표현 행렬 $X \in \mathbb{R}^{m \times p_1}$와 $Y \in \mathbb{R}^{m \times p_2}$가 있을 때, 각각의 그람 행렬(Gram matrix) $K = XX^T$와 $L = YY^T$를 생성한다. CKA 값은 다음과 같이 정규화된 **Hilbert-Schmidt Independence Criterion (HSIC)**으로 정의된다.

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \text{HSIC}(L, L)}}$$

본 논문은 계산 효율성과 성능을 위해 선형 커널(linear kernel)을 사용하며, $\text{HSIC}(K, L) = \text{Tr}(KHLH)$로 단순화하여 계산한다. 여기서 $H$는 centering matrix이다.

### 분석 단위 (Odd vs Even Layers)

ResNet-50의 각 bottleneck layer에서 두 가지 지점의 표현을 추출한다.

- **Odd representations**: residual block 내부의 마지막 Batch Normalization 이후, 덧셈 연산 이전의 표현.
- **Even representations**: residual connection을 통한 덧셈 이후, ReLU 활성화 함수를 거친 최종 표현.

## 📊 Results

### SimCLR 내부 구조 분석

SimCLR의 내부 표현을 CKA로 분석한 결과, 지도학습과 마찬가지로 residual block들이 서로 디커플링(decoupling)되는 경향을 보였다. 이는 SSL이 ResNet 아키텍처를 활용하는 방식이 SL과 유사함을 시사한다.

### SL과 SimCLR의 표현 비교

- **초기 계층 (Early Layers)**: 매우 유사한 표현이 나타난다. 이는 두 방법 모두 Gabor 필터와 같은 공통적인 기본 특징(common primitives)을 학습하기 때문으로 분석된다.
- **잔차 계층 (Odd Layers)**: 초기 계층 이후 유사도가 급격히 낮아진다. 이는 각 방법론이 서로 다른 학습 목적에 따라 입력 데이터를 처리하는 방식이 질적으로 다름을 의미한다.
- **포스트-잔차 계층 (Even Layers)**: 잔차 계층의 불일치에도 불구하고, 누적된 표현인 Even layer들에서는 높은 유사도가 관찰되었다. 즉, 서로 다른 경로를 통해 유사한 최종 표현에 도달하는 구조이다.
- **지연 현상 (Stalling behavior)**: SimCLR는 새로운 Block Group(BG)에 진입할 때 유사도가 일시적으로 정체되었다가 다시 따라잡는 경향을 보인다. 이는 강력한 데이터 증강으로 인해 더 넓은 분포를 압축해야 하는 필요성에서 기인한 것으로 추측된다.

### 최종 계층의 발산 및 성질

최종 Block Group에서 두 방법의 표현은 급격히 발산한다. 이에 대해 다음과 같은 실험을 수행하였다.

1. **Linear Probe Accuracy**: 두 방법 모두 층이 깊어질수록 클래스 구분 가능성(linear separability)이 단조 증가한다. 즉, 구조는 발산하지만 두 모델 모두 클래스 정보를 추출하고 있다.
2. **증강 불변성 (Augmentation Invariance)**: SimCLR는 층이 깊어질수록, 특히 최종 계층에서 매우 높은 증강 불변성을 학습한다. 반면 SL은 이러한 불변성을 암시적으로 학습하지 않는다.
3. **클래스 구조 매핑**: 두 방법 모두 클래스 표현 공간으로의 유사도가 증가하지만, SL은 최종 단계에서 훨씬 더 빠르게 클래스 구조에 피팅된다.

## 🧠 Insights & Discussion

본 논문은 CKA를 통해 학습 방법론에 따른 표현의 차이를 정밀하게 분석하였다는 점에서 강점이 있다. 특히, SSL의 성능이 최종 표현의 형태가 SL과 닮았기 때문이 아니라, **중간 계층에서 학습된 클래스 정보가 풍부한 표현 덕분**이라는 점을 밝혀낸 것이 핵심적인 통찰이다.

**비판적 해석 및 논의사항:**

- **보조 작업 설계의 방향성**: SimCLR가 중간 계층에서 SL의 목적 함수를 암시적으로 학습한다는 점은, 레이블 없는 작업에서도 '심플렉스로의 매핑(mapping to the simplex)'과 같은 유도 편향(inductive bias)을 도입하는 것이 효과적일 수 있음을 시사한다.
- **한계점**: 본 연구는 SimCLR라는 단일 대조 학습 알고리즘만을 대상으로 하였다. BYOL, Barlow Twins 등 다른 SSL 기법들이 동일한 패턴(중간 유사성-최종 발산)을 보이는지는 명시되지 않았으며, 이는 향후 연구 과제로 남아 있다.
- **데이터셋의 규모**: CIFAR-10이라는 상대적으로 작은 데이터셋에서 실험이 진행되었으므로, ImageNet과 같은 거대 데이터셋에서도 동일한 층별 유사성 패턴이 나타날지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 SimCLR(SSL)와 지도학습(SL)이 ResNet-50 내부에서 어떤 표현을 학습하는지 CKA를 통해 분석하였다. 연구 결과, **중간 계층에서는 두 방법이 매우 유사한 클래스 정보 기반 표현을 학습하지만, 최종 계층에서는 SimCLR는 '증강 불변성'을, SL은 '클래스 구조'를 학습하며 서로 발산함**을 발견하였다. 이는 SSL의 성능이 최종 결과물의 유사성이 아닌, 중간 과정에서 학습된 유용한 특징들로부터 온다는 것을 의미하며, 향후 레이블 프리(label-free) 작업 설계 시 중간 특징의 유사성을 높이는 방향으로 연구가 진행될 가능성을 제시한다.
