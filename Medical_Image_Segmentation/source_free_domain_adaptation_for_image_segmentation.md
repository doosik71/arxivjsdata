# SOURCE-FREE DOMAIN ADAPTATION FOR IMAGE SEGMENTATION

Mathilde Bateson, Hoel Kervadec, José Dolz, Hervé Lombaert, Ismail Ben Ayed (2022)

## 🧩 Problem to Solve

본 논문은 이미지 분할(Image Segmentation) 작업에서 소스 도메인(Source Domain)의 데이터에 접근할 수 없는 상황에서의 도메인 적응(Domain Adaptation, DA) 문제를 해결하고자 한다. 일반적인 도메인 적응 기법들은 모델을 적응시키는 과정에서 레이블이 있는 소스 데이터와 레이블이 없는 타겟 데이터를 동시에 사용하는 것을 전제로 한다. 그러나 실제 의료 영상 환경에서는 환자의 개인정보 보호(Privacy concerns), 데이터의 손실 또는 부패, 혹은 실시간 응용을 위한 계산 자원의 제약 등으로 인해 소스 데이터를 타겟 도메인 적응 단계에서 사용할 수 없는 경우가 빈번하게 발생한다.

따라서 본 연구의 목표는 소스 이미지와 레이블 없이, 오직 소스 도메인에서 사전 학습된 모델의 파라미터만을 초기값으로 사용하여 타겟 도메인에 효과적으로 적응시키는 Source-Free Domain Adaptation (SFDA) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 타겟 도메인 데이터에 대해 레이블이 없는 엔트로피 손실(Label-free entropy loss)을 최소화하고, 이를 도메인 불변적인 특성을 가진 클래스 비율 사전 정보(Class-ratio prior)로 가이드하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Source-Free DA 포뮬레이션**: 적응 단계에서 소스 데이터에 전혀 접근하지 않고, 오직 사전 학습된 모델의 가중치만을 사용하여 타겟 도메인에 적응하는 방법론을 제안하였다.
2. **Prior-aware Entropy Minimization**: 섀넌 엔트로피(Shannon entropy)와 타겟 영역의 클래스 비율을 해부학적 사전 정보(Anatomical prior)와 일치시키는 KL 발산(Kullback-Leibler divergence)을 결합한 새로운 손실 함수를 제안하였다.
3. **상호 정보량(Mutual Information)과의 연결**: 제안한 손실 함수가 타겟 이미지와 그 예측 레이블 간의 상호 정보량을 최대화하는 것과 이론적으로 연결되어 있음을 증명하여 방법론의 타당성을 부여하였다.
4. **해부학적 지식의 활용**: 임상 문헌 등에서 얻을 수 있는 해부학적 정보를 통해 클래스 비율 사전 정보를 추정함으로써, 정밀한 레이블 없이도 모델을 효과적으로 가이드할 수 있음을 보였다.

## 📎 Related Works

기존의 도메인 적응 연구들은 크게 다음과 같은 접근 방식을 취해왔다.

- **Adversarial Methods**: GAN과 같은 적대적 학습을 통해 소스와 타겟 도메인 간의 특징 분포를 일치시키는 방식이다. 하지만 설계 구조상 적응 단계에서 소스와 타겟 데이터에 동시에 접근해야 한다는 치명적인 한계가 있다.
- **Self-training 및 Entropy Minimization**: 예측의 확신도를 높이기 위해 엔트로피를 최소화하는 방식이다. 그러나 엔트로피 최소화만으로는 모델이 단일 클래스로 편향되는 Trivial solution(붕괴 현상)에 빠지기 쉬우며, 이를 방지하기 위해 기존 연구들은 소스 데이터의 지도 학습 손실을 함께 사용하였다.
- **Test-time Adaptation (TTA)**: 추론 시점에 적응을 수행하는 방식으로 SFDA 설정과 유사하다. 하지만 기존 TTA 방법들은 소스 학습 단계에서 보조 분기(Auxiliary branches)를 추가하는 등 표준적이지 않은 복잡한 학습 스킴을 요구하는 경우가 많다.

본 논문은 이러한 기존 방식들과 달리, 소스 학습 단계를 수정하지 않으면서도 적응 단계에서 소스 데이터 없이 해부학적 사전 정보를 활용해 Trivial solution 문제를 해결함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인

본 방법론은 두 단계로 구성된다. 첫째, 소스 도메인의 레이블된 데이터를 사용하여 표준적인 지도 학습(Supervised learning)을 통해 모델 $\theta$를 학습시킨다. 둘째, 학습된 모델의 파라미터 $\tilde{\theta}$를 초기값으로 하여 타겟 도메인 데이터만을 이용해 적응 단계(Adaptation phase)를 수행한다.

### 주요 구성 요소 및 손실 함수

**1. 엔트로피 최소화 (Entropy Minimization)**
타겟 도메인 이미지 $I_t$에 대한 예측 결과 $p_t(i, \theta)$의 확신도를 높이기 위해 다음과 같은 가중 섀넌 엔트로피를 최소화한다.
$$\ell_{ent}(p_t(i, \theta)) = -\sum_k \nu_k p_k^t(i, \theta) \log p_k^t(i, \theta)$$
여기서 $\nu_k$는 클래스 불균형을 완화하기 위한 클래스별 가중치이다.

**2. 클래스 비율 사전 정보 (Class-ratio Prior)**
엔트로피 최소화만으로는 특정 클래스로 예측이 쏠리는 현상이 발생하므로, 이미지 내 각 클래스가 차지하는 비율을 규제한다.

- **예측 클래스 비율**: $\hat{\tau}(t, k, \theta) = \frac{1}{|\Omega_t|} \sum_{i \in \Omega_t} p_k^t(i, \theta)$
- **사전 정보 클래스 비율**: $\tau^e(t, k)$ (해부학적 지식 또는 이미지 레벨 태그를 통해 추정)

**3. 최종 손실 함수 (AdaMI)**
본 논문은 예측 비율과 사전 정보 비율 간의 KL 발산을 이용하여 다음과 같은 전체 손실 함수를 정의한다.
$$\min_{\theta} \sum_t \frac{1}{|\Omega_t|} \sum_{i \in \Omega_t} \ell_{ent}(p_t(i, \theta)) + KL(\hat{\tau}(t, \theta, \cdot), \tau^e(t, \cdot))$$
여기서 $KL(\hat{\tau}, \tau^e) = \hat{\tau} \log \frac{\hat{\tau}}{\tau^e}$이다.

### AdaMI와 AdaEnt의 차이 및 이론적 배경

이전 연구(AdaEnt)에서는 KL 발산의 항 순서가 반대($KL(\tau^e, \hat{\tau})$)였으나, 본 논문(AdaMI)에서는 이를 뒤집어 정의하였다. 이는 예측 비율 $\hat{\tau}$가 0에 가까울 때의 그래디언트 역학(Gradient dynamics)이 더 안정적이기 때문이며, 이론적으로는 입력 이미지와 잠재 레이블 간의 상호 정보량(Mutual Information)을 최대화하는 것과 동일함을 보였다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - Spine (IVDM3Seg): Water MRI $\rightarrow$ In-Phase MRI 적응
  - Prostate (NCI-ISBI13): 서로 다른 사이트/장비의 T2-weighted MRI 적응
  - Cardiac (MMWHS): MRI $\rightarrow$ CT 적응
- **평가 지표**: Dice Similarity Coefficient (DSC), Average Symmetric Surface Distance (ASD)
- **비교 대상**: NoAdap(기준점), Oracle(상한선), AdaptSegNet, AdaSource, CDA, TTA, Tent 등

### 정량적 결과

- **성능 향상**: 모든 데이터셋에서 AdaMI는 NoAdap 대비 압도적인 성능 향상을 보였다.
- **SFDA 성능**: 소스 데이터에 접근 가능한 state-of-the-art 방법론(AdaptSegNet 등)과 비교했을 때, 소스 데이터 없이도 상당히 근접한 성능을 달성하였다. 특히 Spine과 Prostate에서는 상위 방법론의 90~95% 수준의 DSC를 기록하였다.
- **Cardiac 적응**: 가장 어려운 MRI $\rightarrow$ CT 적응 작업에서 16개의 적응 기법 중 2위를 기록하며, 소스 데이터 사용 여부와 상관없이 매우 경쟁력 있는 성능을 보였다.

### 분석 결과 (Ablation Study)

- **사전 정보의 부정확성**: 클래스 비율 사전 정보에 $\pm 60\%$의 오차가 있어도 DSC 성능 하락이 최대 15% 수준에 그쳐, 대략적인 해부학적 지식만으로도 충분히 가이드가 가능함을 확인하였다.
- **데이터 효율성**: 타겟 학습 데이터가 단 2명의 피험자분량만 있어도 대부분의 SOTA 방법론과 유사한 수준의 적응 성능을 보였다.
- **약한 지도학습**: 이미지 레벨 태그(Image-level tags)를 제거한 완전 비지도 설정에서도 Baseline보다는 월등히 높은 성능을 유지하였다.

## 🧠 Insights & Discussion

본 논문은 소스 데이터가 없더라도 도메인 불변적인 해부학적 지식(Anatomical knowledge)을 적절히 활용한다면 성공적인 도메인 적응이 가능함을 입증하였다. 특히 다음과 같은 인사이트를 제공한다.

첫째, **소스 데이터의 불필요성**이다. 많은 연구가 소스와 타겟의 분포를 일치시키는 데 집중하지만, 본 연구는 타겟 도메인의 제약 조건(클래스 비율)을 강제하는 것만으로도 충분한 적응이 가능함을 보여주었다. 이는 소스 데이터 접근이 제한적인 의료 환경에서 매우 실용적인 접근이다.

둘째, **해부학적 사전 정보의 강력함**이다. 정밀한 픽셀 단위 레이블이 없더라도, 0차 모멘트(Zero-order shape moments)에 해당하는 클래스 비율 정보가 모델의 붕괴를 막고 정답 영역을 찾는 강력한 가이드가 된다는 점을 확인하였다.

셋째, **계산 효율성**이다. 복잡한 적대적 학습(Adversarial training)이나 보조 네트워크 없이 단일 네트워크의 손실 함수 수정만으로 적응을 수행하므로 계산 비용과 최적화 난이도가 매우 낮다.

한계점으로는 이미지 레벨의 태그 정보가 필요하다는 점이 있으나, 이는 완전한 픽셀 레이블에 비해 획득 비용이 매우 낮으며, 태그가 없는 상황에서도 어느 정도 작동함을 확인하였다.

## 📌 TL;DR

본 논문은 소스 데이터 없이 사전 학습된 모델과 타겟 도메인 데이터만을 사용하는 **Source-Free Domain Adaptation (SFDA)** 방법을 제안한다. 핵심은 타겟 예측의 **엔트로피 최소화**와 **해부학적 클래스 비율 사전 정보**를 결합한 손실 함수(AdaMI)를 사용하는 것이다. 실험 결과, 제안 방법은 소스 데이터를 사용하는 기존 SOTA 방법들과 경쟁 가능한 성능을 보였으며, 특히 의료 영상의 개인정보 보호 문제를 해결하면서도 효율적으로 모델을 적응시킬 수 있는 가능성을 제시하였다.
