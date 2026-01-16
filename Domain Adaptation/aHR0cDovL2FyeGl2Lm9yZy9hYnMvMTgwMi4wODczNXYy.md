# A DIRT-T APPROACH TO UNSUPERVISED DOMAIN ADAPTATION

Rui Shu, Hung H. Bui, Hirokazu Narui, & Stefano Ermon

## 🧩 Problem to Solve

이 논문은 레이블된 데이터가 풍부한 소스 도메인(source domain)에서, 레이블이 부족하거나 없는 타겟 도메인(target domain)을 위한 정확한 모델을 학습하는 **비지도 도메인 적응(unsupervised domain adaptation)** 문제를 다룹니다. 특히, **비보존적(non-conservative)** 설정에 초점을 맞추는데, 이는 단일 분류기(classifier)가 소스 및 타겟 도메인 모두에서 잘 작동한다고 보장할 수 없는 더 어려운 시나리오를 의미합니다.

기존의 도메인 적대적 훈련(Domain Adversarial Training, DANN) 방식은 다음과 같은 두 가지 주요 한계를 가집니다:

1. **제약의 약점:** 특징 추출 함수(feature extraction function)의 용량(capacity)이 높으면, 소스 및 타겟 특징 분포를 맞추려는 제약이 약해질 수 있습니다. 특히 두 도메인의 지지(support)가 서로 겹치지 않는 경우 더욱 그렇습니다.
2. **성능 저하:** 비보존적 도메인 적응 설정에서는 소스 도메인에서 모델이 잘 작동하도록 훈련하는 것이 타겟 도메인의 성능을 저하시킬 수 있습니다.

## ✨ Key Contributions

이 논문은 **클러스터 가정(cluster assumption)**, 즉 "결정 경계는 높은 밀도의 데이터 영역을 가로지르지 않아야 한다"는 원칙을 통해 기존 문제점을 해결하고자 합니다. 주요 기여는 다음과 같습니다:

- **Virtual Adversarial Domain Adaptation (VADA) 모델 제안:** 도메인 적대적 훈련과 클러스터 가정을 위반하는 경우에 대한 페널티 항(virtual adversarial training 및 conditional entropy loss)을 결합하여, 도메인 적대적 훈련의 한계를 개선합니다. 이는 보존적 도메인 적응에서 가설 공간을 추가로 제약하여 성능을 향상시킵니다.
- **Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T) 모델 제안:** VADA 모델을 초기값으로 사용하여, 소스 훈련 신호에서 벗어나 오직 타겟 도메인의 클러스터 가정 위반을 최소화하기 위해 자연 경사(natural gradient) 단계를 반복적으로 적용하여 결정 경계를 정제합니다. 이는 비보존적 도메인 적응에서 타겟 도메인 분류기의 성능을 크게 개선하는 데 중요한 역할을 합니다.
- **최첨단 성능 달성:** 제안된 두 모델의 조합은 숫자 인식(digit recognition), 교통 표지판 인식, Wi-Fi 활동 인식 등 광범위한 도메인 적응 벤치마크에서 기존 최첨단(state-of-the-art) 성능을 크게 뛰어넘는 결과를 보여주었습니다. 특히, 어려운 MNIST→SVHN 적응 작업에서 이전 방법론 대비 20% 이상 성능을 향상시켰습니다.

## 📎 Related Works

- **공변량 이동(Covariate Shift) 보정:** 소스 샘플에 가중치를 재조정하여 타겟 분포와의 불일치를 최소화하는 방식 (Shimodaira, 2000; Mansour et al., 2009).
- **특징 공간 분포 매칭:** 두 분포를 특정 특징 공간에 투영하여 분포 매칭을 유도하는 방식 (Huang et al., 2007; Long et al., 2015).
- **도메인 적대적 훈련(DANN):** 특징 추출기가 소스 및 타겟 도메인 특징 분포를 일치시키도록 훈련하며, 이는 Jensen-Shannon 발산 최소화와 유사 (Ganin & Lempitsky, 2015; Goodfellow et al., 2014).
- **비대칭 삼중 훈련(Asymmetric Tri-training, ATT):** 소스 훈련 분류기로 높은 확신도를 가진 타겟 샘플의 레이블이 정확하다고 가정하여 문제를 해결 (Saito et al., 2017).
- **클러스터 가정:** 결정 경계가 고밀도 영역을 가로지르지 않아야 한다는 가정으로, 준지도 학습(semi-supervised learning)에서 널리 활용됨 (Chapelle & Zien, 2005).
  - **조건부 엔트로피 최소화(Conditional Entropy Minimization):** 분류기가 레이블 없는 데이터에 대해 높은 확신을 가지도록 유도하여 결정 경계를 데이터로부터 멀리 이동 (Grandvalet & Bengio, 2005).
  - **가상 적대적 훈련(Virtual Adversarial Training, VAT):** 분류기의 예측이 인근 지역에서 일관성을 유지하도록 강제하여 모델의 지역적 립시츠(locally-Lipschitz) 제약을 부여 (Miyato et al., 2017).
  - **셀프/템포럴 앙상블링(Self/Temporal-ensembling):** (Laine & Aila, 2016; Tarvainen & Valpola, 2017).

## 🛠️ Methodology

이 논문은 클러스터 가정을 도메인 적응에 적용하여 모델을 개발했습니다.

### VADA (Virtual Adversarial Domain Adaptation)

VADA는 도메인 적대적 훈련에 클러스터 가정을 위반하는 데 대한 페널티를 추가하여 기존 DANN의 한계를 극복합니다.

- **목표 함수:**
  $$ \min\_{\theta} L_y(\theta; D_s) + \lambda_d L_d(\theta; D_s, D_t) + \lambda_s L_v(\theta; D_s) + \lambda_t [L_v(\theta; D_t) + L_c(\theta; D_t)] $$
  - $L_y(\theta; D_s)$: 레이블된 소스 데이터에 대한 교차 엔트로피 손실.
  - $L_d(\theta; D_s, D_t)$: 소스 및 타겟 도메인 특징 분포를 매칭하기 위한 도메인 적대적 손실.
  - $L_c(\theta; D_t) = -E_{x \sim D_t}[h_{\theta}(x)^T \ln h_{\theta}(x)]$: 타겟 데이터에 대한 조건부 엔트로피 최소화. 분류기가 레이블 없는 타겟 데이터에 대해 높은 확신을 갖도록 강제하여 결정 경계를 데이터로부터 멀리 이동시킵니다.
  - $L_v(\theta; D) = E_{x \sim D}[\max_{\Vert r \Vert \le \epsilon} D_{KL}(h_{\theta}(x) \Vert h_{\theta}(x+r))] $: 가상 적대적 훈련(VAT) 손실. 샘플 $x$의 노름 볼(norm-ball) 이웃 내에서 분류기 일관성을 강제하여, 분류기가 훈련 데이터 포인트 근처에서 갑작스럽게 예측을 변경하는 것을 방지하고 지역적 립시츠 제약을 부여합니다. 소스($D_s$)와 타겟($D_t$) 모두에 적용됩니다.
- **$H\Delta H$-거리 최소화:** $\lambda_t > 0$로 설정함으로써 VADA는 높은 타겟-측면 클러스터 가정 위반을 가진 가설을 거부하여 $d_{H\Delta H}$를 줄이고 타겟 일반화 오류에 대한 더 엄격한 상한을 제공합니다.

### DIRT-T (Decision-boundary Iterative Refinement Training with a Teacher)

DIRT-T는 VADA 모델을 초기화로 사용하여 비보존적 도메인 적응 시나리오에 대응합니다.

- **초기화:** VADA 모델에서 학습된 초기 분류기 $h_{\theta_0}$를 사용합니다.
- **반복적 결정 경계 정제:** 소스 훈련 신호를 제거하고, 오직 타겟 도메인에서의 클러스터 가정 위반 손실 $L_t(\theta) = L_v(\theta; D_t) + L_c(\theta; D_t)$을 최소화합니다.
- **자연 경사(Natural Gradient) 사용:** 모델의 파라미터화에 민감한 일반적인 경사 하강법 대신, 분류기의 출력 분포 공간에서의 변화를 측정하는 쿨백-라이블러(Kullback-Leibler, KL) 발산으로 이웃을 정의하여 자연 경사 단계를 취합니다.
  $$ \min*{\Delta\theta} L_t(\theta + \Delta\theta) \quad \text{s.t.} \quad E*{x \sim D*t}[D*{KL}(h*{\theta}(x) \Vert h*{\theta+\Delta\theta}(x))] \le \epsilon $$
    이는 일련의 최적화 문제로 근사화됩니다:
    $$ \min*{\theta_n} \lambda_t L_t(\theta_n) + \beta_t E[D*{KL}(h*{\theta*{n-1}}(x) \Vert h*{\theta_n}(x))] $$
  여기서 $h*{\theta*{n-1}}$는 "교사(teacher)" 모델로, "학생(student)" 모델 $h*{\theta_n}$이 교사 모델에 가까이 머무르면서 클러스터 가정 위반을 줄이도록 안내합니다.
- **비보존적 도메인 적응에 대한 해석:** DIRT-T는 VADA로부터 학습된 결합 분류기에서 더 나은 타겟 도메인 분류기로 전환할 수 있게 합니다. 타겟 분포에 대한 의사 레이블(pseudo-labeling)을 통해 새로운 "소스" 도메인을 구성하는 재귀적 확장으로 해석될 수 있습니다.

## 📊 Results

- **최첨단 성능:** VADA와 DIRT-T는 숫자 인식(MNIST→MNIST-M, SVHN→MNIST, MNIST→SVHN, SYN DIGITS→SVHN), 교통 표지판 인식(SYN SIGNS→GTSRB), 일반 객체 인식(STL-10↔CIFAR-10), Wi-Fi 활동 인식 등 다양한 시각 및 비시각 도메인 적응 벤치마크에서 지속적으로 최첨단 성능을 달성했습니다.
- **VADA의 개선:** VADA는 기존 DANN 및 소스 전용(Source-Only) 모델 대비 상당한 성능 향상을 보였습니다.
- **DIRT-T의 추가 개선:** DIRT-T는 대부분의 작업에서 VADA의 성능을 추가적으로 개선하여, 반복적인 결정 경계 정제의 효과를 입증했습니다. 특히 MNIST→SVHN 같은 어려운 작업에서 인스턴스 정규화 적용 시 76.5%의 정확도를 달성하며 소스 전용 대비 35.6%의 큰 폭의 개선을 보였습니다.
- **인스턴스 정규화(Instance Normalization) 효과:** 인스턴스 정규화를 입력 전처리 단계로 사용했을 때, 대부분의 시각 작업에서 성능 향상이 관찰되었습니다. 이는 모델을 채널-전반적인 픽셀 강도 변화에 불변하게 만들어 $d_{H\Delta H}$를 줄이는 데 도움을 주었습니다.
- **Wi-Fi 활동 인식:** 비시각 도메인 적응 태스크에서도 VADA는 Source-Only 및 DANN 대비 분류 정확도를 크게 향상시켰으나, DIRT-T는 추가적인 개선을 보이지 않았습니다. 이는 VADA가 해당 데이터셋에서 이미 타겟 도메인의 강력한 클러스터링을 달성했기 때문으로 분석됩니다.

## 🧠 Insights & Discussion

- **클러스터 가정의 효과:** VADA와 DIRT-T의 실험적 성공은 심층 신경망을 사용한 비지도 도메인 적응에서 클러스터 가정을 활용하는 것이 매우 효과적임을 강력하게 증명합니다. 결정 경계를 데이터 밀집 영역에서 멀리 떨어뜨리는 것이 일반화 성능에 긍정적인 영향을 미칩니다.
- **가상 적대적 훈련(VAT)의 역할:** VAT는 분류기가 지역적 립시츠 제약을 만족하도록 하여, 조건부 엔트로피 최소화가 의미 있는 방식으로 결정 경계를 데이터로부터 멀리 밀어내는 데 필수적인 역할을 합니다. 대부분의 작업에서 최상의 성능을 달성하는 데 VAT가 필수적임이 ablation 연구를 통해 확인되었습니다.
- **DIRT-T에서 교사 모델 및 자연 경사의 중요성:** DIRT-T에서 KL 발산 항(교사 모델의 역할을 하는)을 통해 모델의 파라미터 공간이 아닌 출력 공간에서 변화를 제어하는 자연 경사 단계를 취하는 것이 중요합니다. 이 항이 없으면 모델이 불안정해지고 타겟 정확도가 급격히 하락하는 경향을 보였습니다. 이는 복잡한 데이터 매니폴드(data manifold)에서 특히 두드러집니다.
- **비보존적 도메인 적응 해결:** DIRT-T는 소스 훈련 신호에서 모델을 분리하고 타겟 도메인 클러스터 가정 위반에만 집중함으로써, 소스 최적 분류기가 타겟 최적 분류기와 일치하지 않는 비보존적 시나리오를 효과적으로 해결합니다.
- **한계점:** DIRT-T는 타겟 훈련 세트가 매우 작을 경우 (예: CIFAR→STL) 조건부 엔트로피 추정의 어려움으로 인해 신뢰성이 떨어질 수 있습니다. 또한, Wi-Fi 인식 작업처럼 VADA가 이미 강력한 클러스터링을 달성한 경우에는 DIRT-T가 추가적인 성능 개선을 가져오지 못할 수 있습니다.

## 📌 TL;DR

**문제:** 비지도, 비보존적 도메인 적응에서 기존 도메인 적대적 훈련(DANN)은 고용량 모델과 도메인 불일치로 인해 한계가 있었습니다.
**방법:** 이 논문은 클러스터 가정을 기반으로 **VADA(Virtual Adversarial Domain Adaptation)**와 **DIRT-T(Decision-boundary Iterative Refinement Training with a Teacher)** 두 가지 모델을 제안합니다. VADA는 DANN에 가상 적대적 훈련(VAT)과 조건부 엔트로피 최소화를 결합하여 클러스터 가정을 강화합니다. DIRT-T는 VADA로 초기화된 모델을 교사로 삼아, 타겟 도메인에 대한 클러스터 가정 위반을 자연 경사(natural gradient)를 통해 반복적으로 정제합니다.
**발견:** VADA와 DIRT-T는 여러 도메인 적응 벤치마크에서 최첨단 성능을 달성했으며, DIRT-T는 VADA의 성능을 더욱 향상시켰습니다. 이는 클러스터 가정과 자연 경사가 도메인 적응, 특히 비보존적 시나리오에서 매우 효과적임을 입증합니다.
