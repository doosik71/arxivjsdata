# Towards Corruption-Agnostic Robust Domain Adaptation

Yifan Xu, Kekai Sheng, Weiming Dong, Baoyuan Wu, Changsheng Xu, and Bao-Gang Hu (2021)

## 🧩 Problem to Solve

본 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 상황에서 테스트 타겟 도메인의 데이터가 예측 불가능한 시각적 부패(Corruption, 예: 노이즈, 블러)를 포함하고 있을 때 발생하는 성능 저하 문제를 해결하고자 한다.

기존의 DA 연구들은 테스트 타겟 도메인이 훈련 타겟 도메인과 동일한 분포(i.i.d.)를 가진다는 이상적인 가정을 전제로 한다. 하지만 실제 웹 이미지와 같은 데이터에는 다양한 부패가 존재하며, 기존의 DA 방법론이나 일반적인 부패 강건성(Corruption Robustness) 방법론(예: AugMix)을 단순히 결합하는 것만으로는 도메인 간의 큰 차이(Domain Discrepancy)와 타겟 도메인의 레이블 부재라는 두 가지 난제를 동시에 해결하기 어렵다.

따라서 본 연구의 목표는 원본 타겟 데이터에 대한 정확도를 유지하면서, 훈련 단계에서 접하지 못한 예측 불가능한 부패에 대해서도 강건한 성능을 보이는 **Corruption-agnostic Robust Domain Adaptation (CRDA)** 과업을 수행하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **도메인 간의 차이(Domain Discrepancy) 정보를 이용하여 예측 불가능한 부패를 모사(Mimic)할 수 있다**는 직관에 기반한다.

1. **Domain Discrepancy Generator (DDG):** 타겟 샘플 주변에서 도메인 차이를 최대화하는 샘플을 생성함으로써, 명시적으로 어떤 부패가 발생할지 알지 못해도 가장 심각한 형태의 부패를 시뮬레이션하는 모듈을 제안한다.
2. **Teacher-Student 기반의 Contrastive Learning:** 타겟 도메인에 레이블이 없으므로, 사전 훈련된 Teacher 모델의 표현력을 활용하여 Student 모델이 DDG로 생성된 부패 샘플과 원본 샘플 간의 특징 거리(Feature Distance)를 좁히도록 학습시키는 체계를 제안한다.

## 📎 Related Works

- **Unsupervised Domain Adaptation (UDA):** 소스 도메인과 타겟 도메인 간의 도메인 불변 특징(Domain-invariant features)을 찾기 위해 MMD(Maximum Mean Discrepancy)나 적대적 학습(Adversarial methods)을 통한 분포 정렬(Distribution Alignment)에 집중해 왔다. 그러나 이러한 방식들은 타겟 데이터의 시각적 부패 가능성을 고려하지 않는다.
- **Corruption Robustness:** ImageNet-C와 같은 벤치마크를 통해 모델의 강건성을 측정하며, AugMix와 같이 일반적인 변환(Transformation)을 조합하여 강건성을 높이는 방식이 주류를 이룬다. 하지만 기존 연구들은 훈련과 테스트 데이터가 동일한 도메인 분포에 있다는 가정하에 진행되며, 도메인 시프트(Domain Shift)가 동반된 상황에서의 강건성은 다루지 않았다.

## 🛠️ Methodology

### 1. Domain Discrepancy Generator (DDG)

DDG의 핵심 가설은 "부패된 타겟 샘플은 원본 타겟 샘플보다 소스 도메인과의 도메인 차이가 더 크다"는 것이다. 따라서 샘플 공간에서 원본 이미지 주변의 $\delta$-이웃 영역 내에서 도메인 차이(Transfer Loss로 측정)를 최대화하는 점을 찾으면, 이것이 가장 심각한 부패 상태를 대변할 수 있다는 논리이다.

수식으로 표현하면, DDG는 다음을 만족하는 샘플 $x_{DDG}^t$를 생성한다.
$$x_{DDG}^t = \arg \max_{\|x_{DDG}^t - x^t\| \le \delta} \ell_{trans}(f(x_{DDG}^t), f(X^s))$$
여기서 $\ell_{trans}$는 MMD나 적대적 손실 함수와 같은 Transfer Loss를 의미한다. 실제 구현에서는 **Projected Gradient Descent (PGD)** 알고리즘을 사용하여 Transfer Loss의 그래디언트를 입력 이미지에 역으로 더함으로써 $\delta$ 범위 내에서 손실을 최대화하는 지점을 찾는다.

### 2. Teacher-Student Warm-up Scheme

타겟 도메인의 레이블이 없기 때문에 강한 제약 조건을 주기 위해 Teacher-Student 구조와 Contrastive Loss를 도입한다.

- **Teacher Model:** 원본 데이터를 잘 표현하도록 사전 훈련된 고정 모델이다.
- **Student Model:** Teacher로부터 지식을 증류(Distill)하며 DDG 샘플에 대한 강건성을 학습하는 모델이다.

**Contrastive Loss ($\ell_{cont}$)**는 동일한 샘플의 원본 버전과 부패 버전(DDG 생성) 간의 특징 벡터 거리를 최소화하도록 설계되었다. 코사인 유사도(Cosine Similarity)를 기반으로 하며, 수식은 다음과 같다.
$$\ell_{sim}(z_i, z_j) = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{z_k \in \mathcal{Z}, z_k \ne z_i} \exp(\text{sim}(z_i, z_k)/\tau)}$$
최종적인 Contrastive Loss $\ell_{cont}^{(DDG)}$는 Student가 추출한 DDG 샘플의 특징 $z_{stu}(x_{DDG}^t)$와 Teacher가 추출한 원본 샘플의 특징 $z_{tea}(x^t)$ 사이의 거리를 좁히는 방향으로 계산된다.

### 3. 전체 학습 목표 (Total Loss)

최종 손실 함수 $\ell_{total}$은 기존 DA 모델의 손실 함수 $\ell_{orig}$와 DDG를 통한 강건성 손실의 가중 합으로 정의된다.
$$\ell_{total} = \ell_{orig}(x^s, x^t, y^s) + \lambda \ell_{cont}^{(DDG)}(x_{DDG}^t, x^t)$$
여기서 $\ell_{orig} = \ell_{cls}(x^s, y^s) + \ell_{trans}(x^s, x^t)$ 이며, $\ell_{cls}$는 소스 도메인의 분류 손실이다.

## 📊 Results

### 실험 설정
- **데이터셋:** Office-Home, Office-31
- **부패 종류:** ImageNet-C의 15가지 부패 유형 (Gaussian Noise, Blur, Fog 등) 및 5단계 심각도
- **지표:** Corruption Error (CE) 및 평균 CE (mCE). mCE가 낮을수록 강건성이 높음을 의미한다.
- **비교 대상:** CDAN+TN, DCAN (DA 베이스라인), AugMix (강건성 베이스라인)

### 주요 결과
- **강건성 향상:** Office-Home 데이터셋에서 DDG를 적용했을 때, 기존 DA 모델 대비 mCE가 크게 낮아졌으며 AugMix보다 더 우수한 일반화 성능을 보였다. (예: CDAN+TN $\to$ DDG 적용 시 mCE 감소)
- **원본 정확도 유지 및 향상:** DDG는 부패에 대한 강건성뿐만 아니라, 깨끗한(Clean) 타겟 데이터에 대한 분류 정확도 또한 유지하거나 오히려 향상시키는 결과를 보였다. 이는 Contrastive Learning을 통한 특징 증류 효과로 해석된다.
- **부패별 분석:** Figure 4에 따르면 DDG는 Pixelate, Glass Blur 등 다양한 부패에서 성능 향상을 보였으나, Contrast 부패에서는 효과가 적었다. 이는 Contrast 부패가 가정한 $\delta$-이웃 범위를 벗어나는 큰 시프트를 일으키기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 통찰
- **도메인 정보의 활용:** 단순한 데이터 증강(Augmentation) 대신 도메인 간의 불일치(Discrepancy)라는 DA의 고유 정보를 이용하여 부패를 모사했다는 점이 독창적이다.
- **계산 효율성:** 이론적 분석(Proposition 1)을 통해 $\delta$-이웃의 경계점(edge points)만 정렬해도 강건성을 얻을 수 있음을 보였고, 이를 통해 PGD의 업데이트 단계 $n$을 2회로 매우 낮게 설정해도 충분한 성능이 나옴을 입증하였다.
- **Order-invariant Representation:** 네트워크가 부패의 심각도 수준에 따라 특징 공간에서의 거리가 일관되게 증가하는 '순서 불변 표현'을 학습한다는 점을 발견하여, DDG의 이론적 근거(Assumption 2)를 뒷받침하였다.

### 한계 및 비판적 해석
- **$\delta$-범위의 제약:** 본 논문의 방법론은 부패된 샘플이 원본의 $\delta$-근방에 존재한다는 가정(Assumption 1)에 의존한다. 실험 결과에서 나타났듯, 이 범위를 크게 벗어나는 부패(예: Contrast)에 대해서는 대응 능력이 떨어진다.
- **Lower Bound와의 간극:** 테스트 부패를 미리 알고 훈련했을 때의 성능(Lower Bound)과 비교했을 때 여전히 큰 격차가 존재한다. 이는 "알 수 없는 부패"를 모사하는 것만으로는 완벽한 강건성을 달성하기 어렵다는 것을 시사한다.

## 📌 TL;DR

본 논문은 도메인 적응(DA) 상황에서 타겟 데이터에 발생할 수 있는 예측 불가능한 시각적 부패에 대응하는 **CRDA(Corruption-agnostic Robust Domain Adaptation)**라는 새로운 과업을 정의하고, 이를 해결하기 위한 **DDG(Domain Discrepancy Generator)** 모듈을 제안하였다. DDG는 도메인 간 차이를 최대화하는 샘플을 생성하여 부패를 모사하며, 이를 Teacher-Student 구조의 Contrastive Learning으로 학습시켜 원본 성능 저하 없이 강건성을 확보하였다. 이 연구는 DA 모델이 실제 환경의 노이즈와 오염에 대응하는 능력을 키우는 데 중요한 방향성을 제시한다.