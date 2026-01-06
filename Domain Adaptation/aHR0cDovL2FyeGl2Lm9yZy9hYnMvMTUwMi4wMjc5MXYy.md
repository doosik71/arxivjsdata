# Learning Transferable Features with Deep Adaptation Networks

Mingsheng Long, Yue Cao, Jianmin Wang, Michael I. Jordan

## 🧩 Problem to Solve

심층 신경망(DNN)은 새로운 작업에 잘 일반화되는 전이 가능한(transferable) 특징을 학습할 수 있지만, 네트워크의 후반 계층으로 갈수록 특징은 일반적인 것에서 특정적인 것으로 변화하며, 도메인 간의 불일치(domain discrepancy)가 커짐에 따라 전이성이 크게 저하됩니다. 특히 작업별(task-specific) 계층에서 이러한 특징 전이성 감소는 심화되며, 이는 대상 도메인(target domain)에서의 예측 모델 성능 저하로 이어집니다. 이 연구는 이러한 문제를 해결하기 위해 후반 계층에서 데이터셋 편향(dataset bias)을 공식적으로 줄이고 특징 전이성을 높이는 방법을 제안합니다.

## ✨ Key Contributions

- **새로운 심층 적응 네트워크(DAN) 아키텍처 제안**: 도메인 적응 시나리오에 특화된 심층 컨볼루션 신경망(CNN)을 일반화한 DAN 아키텍처를 제안합니다.
- **다중 계층 적응(Multi-layer Adaptation)**: $fc6$부터 $fc8$까지 모든 작업별(task-specific) 계층의 은닉 표현(hidden representations)에 대해 도메인 불일치를 명시적으로 줄임으로써 "심층 적응(deep adaptation)"의 이점을 제공합니다.
- **다중 커널 적응(Multi-kernel Adaptation)**: 재현 커널 힐베르트 공간(RKHS)에서 평균 임베딩(mean embeddings)을 매칭할 때 최적의 다중 커널 선택 방법을 사용하여 적응 효과를 크게 향상시키며, 통계적 보증(statistical guarantees)을 가진 편향 없는 심층 특징을 생성합니다.
- **선형 시간 알고리즘**: 커널 평균 임베딩의 편향 없는 추정(unbiased estimate)을 통해 계산 비용을 기존 $O(n^2)$에서 $O(n)$으로 줄여, 대규모 데이터셋 학습에 대한 확장성(scalability)을 확보했습니다.
- **최첨단 성능 달성**: 표준 도메인 적응 벤치마크에서 기존 방법들 대비 최첨단(state-of-the-art) 이미지 분류 오류율을 달성합니다.

## 📎 Related Works

- **전이 학습(Transfer Learning)**: 도메인 불일치를 고려하여 다른 도메인이나 작업 간의 모델을 구축하는 연구 (Pan & Yang, 2010).
- **얕은 특징 기반 도메인 적응**: 도메인 불일치를 줄이는 새로운 얕은 표현 모델 학습에 초점 (Pan et al., 2011; Long et al., 2013).
- **심층 신경망의 전이성**: 심층 신경망이 전이 가능한 특징을 학습할 수 있음을 보여주는 연구 (Glorot et al., 2011; Donahue et al., 2014; Yosinski et al., 2014).
- **최대 평균 불일치(MMD) 기반 방법**: 두 분포 간의 거리를 측정하여 도메인 불일치를 줄이는 방법 (Gretton et al., 2012a;b).
- **DDC (Deep Domain Confusion)**: 심층 CNN에 적응 계층을 추가하고 단일 커널 MMD로 정규화하여 도메인 불변 표현을 학습하는 이전 연구 (Tzeng et al., 2014). 이 방법은 단일 계층만 적응하고 MMD 계산 비용이 비효율적이라는 한계가 있습니다.
- **도메인 적응 이론**: 대상 도메인 위험을 원본 도메인 위험과 도메인 불일치 메트릭으로 경계 짓는 이론적 분석 (Ben-David et al., 2010).

## 🛠️ Methodology

DAN은 ImageNet으로 사전 학습된 AlexNet 모델을 기반으로 하며, 다음과 같은 단계로 특징 전이성을 강화합니다.

1. **네트워크 아키텍처**:

   - $conv1$~$conv3$ 계층은 일반적인 특징을 추출하므로 가중치를 **고정(frozen)**합니다.
   - $conv4$~$conv5$ 계층은 약간 도메인 편향적(domain-biased)이므로 **미세 조정(fine-tuning)**합니다.
   - $fc6$~$fc8$ 계층은 작업별(task-specific) 특징을 학습하므로 전이성이 낮아, **MK-MMD(Multi-kernel Maximum Mean Discrepancy)**를 통해 적응(adapt)합니다.

2. **손실 함수**:

   - 원래 CNN의 교차 엔트로피 손실(cross-entropy loss)과 $fc6$~$fc8$ 계층의 도메인 불일치를 줄이기 위한 다중 계층 MK-MMD 정규화 항을 결합합니다:
     $$ \min*{\Theta} \frac{1}{n_a} \sum*{i=1}^{n*a} J(\theta(x*{a*i}), y*{a*i}) + \lambda \sum*{\ell=l_1}^{l_2} d_k^2(D^\ell_s, D^\ell_t) $$
        여기서 $J$는 교차 엔트로피 손실, $\theta$는 CNN의 예측 함수, $\lambda$는 페널티 파라미터, $D^\ell_s$와 $D^\ell_t$는 $\ell$번째 계층의 원본 및 대상 도메인 표현입니다.

3. **MK-MMD 계산**:

   - 두 분포 $p, q$ 간의 Squared MK-MMD는 RKHS에서 평균 임베딩 거리로 정의됩니다: $d_k^2(p, q) = \left\| E_p[\phi(x_s)] - E_q[\phi(x_t)] \right\|_{H_k}^2$.
   - 커널 $k$는 $m$개의 양의 반정부호(PSD) 커널 $k_u$의 볼록 조합 $k = \sum_{u=1}^m \beta_u k_u$으로 구성됩니다.
   - 기존 $O(n^2)$ 계산 복잡성을 해결하기 위해 선형 시간 $O(n)$의 편향 없는 MMD 추정기(unbiased estimator)를 사용합니다.

4. **최적화 알고리즘**:
   - **$\Theta$ 학습**: 미니 배치 확률적 경사 하강법(SGD)을 사용하여 CNN 파라미터 $\Theta$를 업데이트합니다. 이때 MMD 정규화 항의 기울기는 선형 시간으로 효율적으로 계산됩니다.
   - **$\beta$ 학습**: 각 계층의 MK-MMD 파라미터 $\beta$는 MMD의 검정력(test power)을 최대화하고 제2종 오류(Type II error)를 최소화하는 2차 계획법(Quadratic Program, QP)을 통해 최적화됩니다: $\min_{\beta \text{ s.t. } d^T\beta=1, \beta>0} \beta^T(Q+\epsilon I)\beta$.
   - $\Theta$와 $\beta$는 이터레이션(iteration) 방식으로 교대로 업데이트됩니다.

## 📊 Results

- **Office-31 및 Office-10 + Caltech-10 데이터셋**에서 표준 도메인 적응 벤치마크 테스트 결과, DAN은 모든 이전 방법(TCA, GFK, CNN, LapCNN, DDC)을 능가하는 최첨단 성능을 보였습니다.
- **다중 커널 및 다중 계층의 효과**:
  - 단일 계층 적응 DAN ($DAN_7$, $DAN_8$)은 DDC보다 우수했는데, 이는 다중 커널 MMD가 단일 커널 MMD보다 도메인 불일치를 더 효과적으로 해결함을 시사합니다.
  - 단일 커널을 사용한 다중 계층 적응 DAN ($DAN_{SK}$) 역시 DDC보다 높은 정확도를 달성하여, 심층 아키텍처의 분포 적응 능력을 확인했습니다.
  - 완전한 DAN 모델은 다중 계층 적응과 다중 커널 MMD를 결합하여 가장 좋은 성능을 보였습니다.
- **특징 시각화 (t-SNE)**: DAN으로 학습된 특징은 DDC 특징보다 대상 도메인 내의 클래스 구분이 더 명확하고, 원본 도메인과 대상 도메인 간의 카테고리 정렬(category alignment)이 더 잘 이루어졌음을 보여주었습니다.
- **A-거리 분석**: DAN 특징의 A-거리는 CNN 특징보다 작아서 더 나은 전이성을 보장했습니다. 이는 DAN이 확대된 도메인 불일치를 효과적으로 줄임을 의미합니다.
- **파라미터 $\lambda$ 민감도**: $\lambda$ 값의 변화에 따라 분류 정확도가 종 모양 곡선(bell-shaped curve)을 보였는데, 이는 심층 특징 학습과 분포 불일치 적응 간의 최적의 균형점이 특징 전이성을 향상시키는 데 중요함을 시사합니다.

## 🧠 Insights & Discussion

- 이 연구는 심층 신경망의 고수준 계층에서 추출되는 작업별 특징의 전이성이 저하되는 문제를 성공적으로 해결했습니다. 일반적인 특징은 전이성이 좋지만, 특정적인 특징은 도메인 불일치를 효과적으로 연결하지 못한다는 가설을 입증했습니다.
- 다중 계층 표현에 대한 평균 임베딩 매칭(mean-embedding matching)을 통해 특징 전이성을 크게 향상시킬 수 있음을 보여주었습니다.
- 최적의 다중 커널 선택 전략과 커널 평균 임베딩의 선형 시간 편향 없는 추정치는 대규모 데이터셋에 대한 딥러닝에서 매우 중요한 확장성과 효율성을 제공합니다.
- DAN은 도메인 적응을 위한 심층 모델 설계에 있어 다중 계층 및 다중 커널 적응의 중요성을 강조합니다.
- 향후 연구에서는 네트워크의 일반성과 특정성 경계를 결정하는 원칙적인 방법과 컨볼루션 계층에 대한 분포 적응 적용 가능성을 탐색할 수 있습니다.

## 📌 TL;DR

DAN(Deep Adaptation Network)은 심층 신경망의 후반 계층에서 떨어지는 특징 전이성을 개선하기 위한 새로운 아키텍처입니다. 작업별 계층($fc6$-$fc8$)에 대해 다중 커널 MMD(Maximum Mean Discrepancy)를 사용하여 도메인 불일치를 명시적으로 줄이는 다중 계층 적응 기법을 적용하며, 커널 평균 임베딩의 선형 시간 추정으로 대규모 학습의 효율성을 높였습니다. 결과적으로 DAN은 기존 도메인 적응 방법들을 능가하는 최첨단 성능을 달성하며, 심층 신경망의 전이 학습 능력을 크게 향상시켰습니다.
