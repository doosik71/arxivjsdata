# Semi-supervised Left Atrium Segmentation with Mutual Consistency Training

Yicheng Wu, Minfeng Xu, Zongyuan Ge, Jianfei Cai, and Lei Zhang

---

## 🧩 Problem to Solve

의료 영상 분할 작업은 정확한 진단을 위해 필수적이지만, 수많은 고밀도(densely annotated) 레이블링된 데이터의 수집은 매우 노동 집약적이고 시간이 많이 소요됩니다. 반지도 학습(Semi-supervised learning)은 이러한 데이터 레이블링 부담을 줄이는 데 유용하지만, 기존 대부분의 방법들은 모델의 불확실성을 최소화하는 데 중요한 정보를 담고 있는 도전적 영역(예: 작은 가지, 흐릿한 경계)의 중요성을 과소평가합니다. 적은 레이블링된 데이터로 학습된 딥러닝 모델은 이러한 복잡하거나 모호한 영역에서 모호한 예측(ambiguous predictions)을 보이는 경향이 있습니다.

## ✨ Key Contributions

* **모델 기반 불확실성 정보 탐색:** 학습 과정에서 레이블링되지 않은 도전적 영역(challenging regions)의 중요성을 강조하기 위해 모델 기반의 불확실성 정보(uncertainty information)를 활용합니다.
* **순환 의사 레이블(Cycled Pseudo Label) 기법 설계:** 상호 일관성(mutual consistency)을 장려하여 모델 학습을 촉진하는 새로운 순환 의사 레이블 기법을 제안합니다.
* **최신 성능 달성:** 제안된 MC-Net은 LA(Left Atrium) 데이터베이스에서 반지도 좌심방 분할(left atrium segmentation) 작업에서 새로운 최신 성능(state-of-the-art)을 달성했습니다.

## 📎 Related Works

* **일관성 정규화(Consistency Regularization):**
  * 데이터 증강(perturbations)을 통해 모델이 불변적인 결과를 출력하도록 제약하는 방법 (Sohn et al. [11]).
  * 의사 레이블(pseudo labels)을 사용하여 낮은 밀도 분리(low-density separation)를 장려하는 엔트로피 정규화(entropy regularization) 방법 (Lee et al. [5]).
* **의료 영상 반지도 분할:**
  * 교사-학생(teacher-student) 모델을 통해 좌심방을 분할하는 방법 (Yu et al. [17]).
  * 레이블링된 데이터와 언레이블된 데이터의 특징 공간을 가깝게 만들기 위해 적대적 손실(adversarial loss)을 도입하는 방법 (Li et al. [6]).
  * 의미 분할(semantic segmentation)과 모양 회귀(shape regression) 간의 관계를 연구하여 언레이블된 데이터를 활용하는 방법 (Luo et al. [7]).
  * 어텐션 메커니즘(attention mechanisms)을 사용하여 레이블링된 데이터와 언레이블된 데이터 간의 의미론적 유사성을 계산하는 방법 (Xie et al. [15]).
  * 공동 훈련(co-training) 프레임워크와 적대적 손실을 활용하는 방법 (Fang et al. [1]).
* **기존 방법의 한계:** 대부분의 딥러닝 모델들은 성능 향상을 위해 추가적인 구성 요소가 필요하거나, 훈련 과정에서 도전적 영역(예: 작은 가지 또는 표적 주변의 접착 경계)의 중요성을 과소평가합니다.

## 🛠️ Methodology

본 논문에서는 `MC-Net (Mutual Consistency Network)`을 제안하며, 이는 `하나의 인코더`와 `두 개의 약간 다른 디코더`로 구성됩니다. 두 디코더의 예측 불일치를 활용하여 모델의 불확실성을 줄이고, 이를 순환 의사 레이블 기법을 통해 비지도 손실로 변환합니다.

1. **모델 아키텍처:**
    * `V-Net`을 기반으로 하며, `인코더($\Theta_e$)`에서 추출된 심층 특징(`F_e`)을 두 개의 디코더가 공유합니다.
    * `디코더 $\Theta_{dA}$`: 기존 V-Net처럼 전치 컨볼루션(transposed convolution)을 사용하여 업샘플링(up-sampling)합니다.
    * `디코더 $\Theta_{dB}$`: 보조 분류기(auxiliary classifier)로서 삼선형 보간법(tri-linear interpolation)을 사용하여 특징 맵을 확장합니다.
    * 두 디코더의 약간 다른 설계는 세그멘테이션 모델의 다양성을 증가시켜 과적합을 줄이고 성능을 향상시킵니다.
    * 모델 기반의 `인식론적 불확실성(epistemic uncertainty)`은 두 디코더의 출력 `P_A`와 `P_B` 간의 불일치로 근사화됩니다. 이는 Monte Carlo Dropout 방식보다 계산 비용이 적습니다.

2. **순환 의사 레이블(Cycled Pseudo Label):**
    * **소프트 의사 레이블 생성:** 예측 확률 출력 `P_A`와 `P_B`를 샤프닝 함수(sharpening function)를 통해 `소프트 의사 레이블($sPL_A$, $sPL_B$)`로 변환합니다. 샤프닝 함수는 다음과 같이 정의됩니다:
        $$ sPL = \frac{P^{1/T}}{P^{1/T} + (1-P)^{1/T}} $$
        여기서 `T`는 샤프닝 온도를 제어하는 상수입니다. 소프트 의사 레이블은 엔트로피 정규화에 기여하며 오분류된 훈련 데이터의 영향을 완화합니다.
    * **상호 일관성 훈련:** `sPL_A`는 `P_B`를 지도하고, `sPL_B`는 `P_A`를 지도하여 `상호 일관성(mutual consistency)`을 달성합니다. 이를 통해 두 디코더는 서로에게서 배우며 `엔드-투-엔드(end-to-end)` 방식으로 훈련됩니다. 이 과정은 모델이 레이블링되지 않은 도전적이고 불확실한 영역에 더 많은 관심을 기울이도록 유도합니다.
    * **총 손실 함수:** `MC-Net`은 세그멘테이션 손실 `L_{seg}`와 일관성 손실 `L_c`의 가중합으로 훈련됩니다:
        $$ \text{loss} = \underbrace{\text{Dice}(P_A, Y) + \text{Dice}(P_B, Y)}_{L_{seg}} + \lambda \times \underbrace{(L_2(P_A, sPL_B) + L_2(P_B, sPL_A))}_{L_c} $$
        여기서 `Dice`는 Dice 손실, `L_2`는 MSE(Mean Squared Error) 손실, `Y`는 ground truth입니다. `L_{seg}`는 레이블링된 데이터에만 적용되며, `L_c`는 비지도 학습 손실로 모든 훈련 데이터에 적용됩니다.

## 📊 Results

* **데이터베이스:** 2018 Atrial Segmentation Challenge의 LA(Left Atrium) 데이터베이스에서 평가되었습니다. 100개의 MR 영상 중 80개는 훈련용, 20개는 검증용으로 사용되었습니다.
* **성능 우수성:**
  * MC-Net은 LA 데이터베이스에서 6가지 최신 반지도 학습 방법들을 능가하며, 모든 반지도 설정(10% 및 20% 레이블링된 데이터)에서 우수한 성능을 보였습니다.
  * 특히, 20%의 레이블링된 데이터만으로도 MC-Net은 Dice 점수 90.34%를 달성하여, 100% 레이블링된 데이터로 학습된 V-Net의 성능(Dice 91.14%)에 필적하는 결과를 보였습니다.
* **시각적 개선:** MC-Net은 3D 또는 2D 뷰에서 기존 최신 방법들보다 더 완전한 좌심방 분할 결과를 생성했으며, 특히 도전적 영역(예: Figure 3의 노란색 원)에서 더 나은 결과를 보이고 대부분의 고립된(isolated) 영역을 제거했습니다.
* **어블레이션 연구(Ablation Study):**
  * 두 개의 약간 다른 디코더(`V2d-Net`)를 사용하는 것이 동일한 디코더(`V2-Net`)를 사용하는 것보다 더 나은 결과를 생성하여 모델 다양성 증가의 효과를 입증했습니다.
  * `소프트 의사 레이블(sPL)`을 사용하여 일관된 결과를 장려하는 것과 `순환 의사 레이블(CPL)`을 사용하여 엔트로피 정규화를 적용하는 것이 추가적인 성능 향상에 기여했습니다.

## 🧠 Insights & Discussion

* 이 연구는 레이블링되지 않은 데이터, 특히 모델이 불확실한 예측을 하는 도전적 영역이 반지도 학습에서 매우 중요하며, 이러한 영역을 강조하여 모델을 훈련하는 것이 효과적임을 보여줍니다.
* 제안된 상호 일관성 훈련 방식은 모델이 일관적이고 낮은 엔트로피의 예측을 하도록 유도함으로써, 언레이블된 도전적 영역에서 더욱 일반화된 특징 표현(generalized feature representation)을 학습할 수 있게 합니다.
* MC-Net은 기존 SOTA 방법들을 뛰어넘는 성능을 달성하여 의료 영상 분할 분야에서의 실용적 가치를 입증했습니다.
* 제안된 MC-Net은 다른 모양 제약(shape-constrained) 모델들과 쉽게 결합하여 세그멘테이션 결과를 더욱 향상시킬 수 있는 잠재력을 가집니다.

## 📌 TL;DR

의료 영상 분할은 레이블링된 데이터 부족 문제에 직면해 있습니다. 본 논문은 언레이블된 도전적 영역의 중요성을 강조하기 위해 `상호 일관성 네트워크(MC-Net)`를 제안합니다. MC-Net은 `하나의 인코더`와 `두 개의 약간 다른 디코더`로 구성되며, `순환 의사 레이블(cycled pseudo label)` 기법을 통해 두 디코더의 예측 불일치를 비지도 손실로 활용하여 상호 일관성 학습을 유도합니다. 이 접근 방식은 모델이 일관되고 낮은 엔트로피의 예측을 생성하도록 촉진하여 언레이블된 도전적 영역에서 일반화된 특징을 학습하게 합니다. 실험 결과, MC-Net은 적은 레이블 데이터만으로도 `LA 데이터베이스`에서 기존 최신 반지도 학습 방법들을 능가하는 `최신 성능`을 달성했습니다.
