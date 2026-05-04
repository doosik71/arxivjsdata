# Representation Learning with Autoencoders for Electronic Health Records: A Comparative Study

Najibesadat Sadati, Milad Zafar Nezhad, Ratna Babu Chinnam, Dongxiao Zhu (2019)

## 🧩 Problem to Solve

전자 건강 기록(Electronic Health Records, EHR)은 최근 급격히 증가하고 있으며, 이를 활용한 헬스케어 연구의 기회는 넓어지고 있다. 그러나 EHR 데이터는 기본적으로 고차원(high-dimensional)이며, 희소성(sparse)이 높고 구조가 복잡하다는 특성을 가진다. 특히 임상 데이터에 레이블(label)을 지정하는 작업은 비용이 많이 들고 시간이 오래 걸리며 매우 어렵다.

본 논문은 이러한 고차원적이고 복잡한 EHR 데이터에서 유의미한 통찰을 얻기 위해, 레이블이 없는 데이터가 풍부한 상황에서도 효과적으로 작동할 수 있는 특징 표현 학습(Representation Learning) 방법을 탐구한다. 구체적으로는 다양한 딥러닝 기반의 특징 표현 기법들이 데이터셋의 규모(소규모 vs 대규모)에 따라 예측 성능에 어떤 영향을 미치는지 비교 분석하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 비지도 학습(Unsupervised Learning)을 통해 EHR 데이터의 고차원 특징을 추상화된 저차원 표현으로 변환하고, 그 위에 지도 학습(Supervised Learning) 모델을 얹어 예측 성능을 높이는 세미-지도 학습(Semi-supervised Learning) 프레임워크인 DIP(Deep Integrated Predictive) 접근 방식을 제안하는 것이다.

주요 기여 사항은 다음과 같다.

1. **DIP 프레임워크 제안**: 전처리, 딥러닝 기반 특징 표현 학습, 지도 학습 예측으로 이어지는 통합 파이프라인을 구축하였다.
2. **다양한 딥러닝 아키텍처 비교**: Stacked Sparse Autoencoder (SSAE), Deep Belief Network (DBN), Adversarial Autoencoder (AAE), Variational Autoencoder (VAE) 네 가지 모델의 성능을 비교 분석하였다.
3. **데이터 규모에 따른 가이드라인 제공**: 데이터셋의 크기에 따라 최적의 특징 표현 기법이 달라짐을 입증하여, 실제 적용 시 유용한 실무적 지침을 제공하였다.

## 📎 Related Works

### 기존의 특징 학습 방법

- **Shallow Feature Learning**: PCA(주성분 분석)와 같은 선형 방식이나 Isomap, LLE와 같은 비선형 매니폴드 학습 방식이 사용되었다. 이러한 방식은 계산 효율성이 좋고 해석 가능성이 높지만, 복잡한 EHR 데이터의 고차원적인 추상적 특징을 추출하는 데 한계가 있다.
- **Deep Supervised Learning**: CNN이나 RNN을 사용하여 특징을 직접 학습하는 방식이다. 하지만 이는 대규모의 레이블된 데이터가 필요하며, 의료 데이터 특성상 레이블 확보가 어렵다는 치명적인 단점이 있다.

### 차별점

본 연구는 기존의 'Deep Patient'나 'Doctor AI'와 같이 비지도 학습을 선행하는 접근 방식을 따르지만, 단순히 한 가지 모델을 사용하는 것에 그치지 않고 다양한 생성 모델(VAE, AAE)과 일반 오토인코더(SSAE, DBN)를 실제 의료 데이터셋 규모에 따라 정밀하게 비교했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 (DIP Approach)

DIP 접근 방식은 다음의 세 단계로 구성된다.

- **Step A (Preprocessing)**: 결측치 처리(Imputation)와 이상치 제거를 수행하며, 범주형 및 텍스트 변수는 GloVe 알고리즘을 이용한 Word Embedding을 통해 벡터로 변환한다.
- **Step B (Feature Representation)**: 비지도 학습 기반의 딥러닝 모델을 사용하여 입력 데이터를 고차원 추상화된 특징 맵으로 변환한다.
- **Step C (Supervised Learning)**: 추출된 특징을 입력으로 하여 Random Forests, SVM, Lasso와 같은 지도 학습 모델을 통해 최종 타겟 값을 예측한다.

### 특징 표현을 위한 딥러닝 아키텍처 및 수식

#### 1. Stacked Sparse Autoencoder (SSAE)

입력 $x$를 인코딩하여 잠재 표현을 만들고, 이를 다시 디코딩하여 $x'$를 복원하는 구조이다. 기본 손실 함수는 다음과 같다.
$$\text{Loss}(x,x') = \|x-x'\| = \|x-f(W'(f(Wx+b)) +b')\|$$
여기서 $f$는 활성화 함수이다. SSAE는 과적합을 방지하고 유의미한 특징을 찾기 위해 희소성 제약(Sparsity Constraint)을 추가하며, $L1$ 정규화나 KL-Divergence를 손실 함수에 더한다.
$$\text{Loss}(x,x') + \lambda \sum_{i} |a_i^{(h)}| \quad \text{or} \quad \text{Loss}(x,x') + \lambda \sum_{j} KL(\rho \| \bar{\rho}_j)$$

#### 2. Deep Belief Network (DBN)

여러 개의 RBM(Restricted Boltzmann Machine)을 쌓아 올린 구조이며, 탐욕적 층별 학습(Greedy layer-wise training) 방식을 사용한다. 가시층과 은닉층 사이의 결합 확률 분포는 다음과 같이 정의된다.
$$P(x,h_1,...,h_l) = \prod_{k=0}^{l-2} P(h_k|h_{k+1})P(h_{l-1}h_l)$$

#### 3. Variational Autoencoder (VAE)

잠재 변수 $z$의 분포를 학습하는 확률적 생성 모델이다. 입력 데이터의 분포를 다음과 같은 베이징 접근법으로 모델링한다.
$$p(x) = \int p(x,z)dz = \int p(x|z)p(z)dz$$
계산 가능하게 만들기 위해 변분 하한(Variational Lower Bound)을 최대화하며, KL-Divergence를 사용하여 잠재 공간을 정규화한다.
$$\mathbb{E}_{q_\phi(z|x)}[\log p(x|z)] - D(q(z|x)\|p(z))$$

#### 4. Adversarial Autoencoder (AAE)

GAN의 적대적 학습 구조를 오토인코더에 접목한 모델이다. 생성자(G)는 잠재 변수를 데이터 공간으로 매핑하고, 판별자(D)는 이 샘플이 실제 분포에서 왔는지 생성자가 만들었는지 구분한다.
$$\min_{G} \max_{D} \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1-D(G(z)))]$$

## 📊 Results

### 실험 설정

- **데이터셋**:
  - 소규모 데이터: DMC 데이터셋 (심혈관 질환 위험 예측, 91명 환자, 172개 속성).
  - 대규모 데이터: eICU 데이터셋 (Cardiac-ICU 및 Neuro-ICU 환자의 입원 기간(LOS) 예측, 각각 약 7,000~8,000개 레코드, 150개 이상의 속성).
- **평가 지표**: Root Mean Squared Error (RMSE).
- **비교 대상**: 원본 데이터(Original)를 사용한 예측 및 기존의 의료 지표인 APACHE 점수 기반 예측.

### 주요 결과

1. **소규모 데이터셋 (DMC)**:
    - 특징 표현 학습을 적용한 모든 모델이 원본 데이터를 사용했을 때보다 낮은 RMSE를 기록하며 성능이 향상되었다.
    - 특히 **SSAE**와 Random Forest의 조합이 가장 우수한 성능(RMSE 6.89)을 보였다.

2. **대규모 데이터셋 (eICU)**:
    - **VAE**가 다른 모든 모델을 압도하는 가장 뛰어난 예측 성능을 보여주었다.
    - AAE 역시 우수한 성능을 보였으나 VAE가 가장 낮고 안정적인 RMSE를 기록하였다.
    - 기존의 APACHE 방식보다 딥러닝 기반 특징 표현 방식이 훨씬 더 정확한 예측력을 가짐을 확인하였다.

## 🧠 Insights & Discussion

### 데이터 규모에 따른 모델 선택의 근거

본 논문은 왜 데이터 규모에 따라 성능 차이가 발생하는지를 분석하였다.

- **SSAE의 강점 (소규모)**: 소규모 데이터에서는 데이터의 양이 적어 과적합 위험이 크다. SSAE는 희소성 정규화(Sparsity Regularization)를 통해 모델의 복잡도를 제어하므로 소규모 데이터에서 더 일반화된 성능을 낼 수 있다.
- **VAE의 강점 (대규모)**: 대규모 데이터에서는 데이터의 내재된 실제 분포(True Distribution)를 학습하는 것이 중요하다. VAE는 확률적 분포를 학습하는 생성 모델이므로, 데이터가 충분할 때 훨씬 더 정교하고 강건한 특징 표현을 생성할 수 있다.

### 한계 및 향후 과제

본 연구는 특징 표현 학습의 효과를 입증하였으나, 딥러닝 모델의 특성상 하이퍼파라미터 튜닝에 많은 시간이 소요되고 내부 동작을 해석하는 것이 어렵다는 한계가 있다. 저자들은 향후 연구에서 샘플 수($n$)와 특징 수($p$)의 조합에 따른 네 가지 시나리오(large $n$/large $p$, large $n$/small $p$ 등)를 설정하여 더 정밀한 가이드라인을 제시할 필요가 있다고 언급한다.

## 📌 TL;DR

이 논문은 고차원적이고 희소한 EHR 데이터의 예측 성능을 높이기 위해 비지도 학습 기반의 특징 표현 학습과 지도 학습을 결합한 DIP 프레임워크를 제안한다. 실험 결과, **소규모 데이터셋에서는 정규화 능력이 뛰어난 SSAE**가, **대규모 데이터셋에서는 분포 학습 능력이 탁월한 VAE**가 가장 효과적임을 밝혀냈다. 이 결과는 의료 데이터의 규모에 따라 적절한 딥러닝 아키텍처를 선택해야 한다는 실무적인 가이드라인을 제공하며, 향후 정밀 의료 및 환자 위험 예측 시스템 구축에 중요한 기초 자료가 될 것으로 보인다.
