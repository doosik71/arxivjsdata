# SegMatch: Semi-supervised surgical instrument segmentation

Meng Wei, Charlie Budd, Luis C. Garcia-Peraza-Herrera, Reuben Dorent, Miaojing Shi, and Tom Vercauteren (2025)

## 🧩 Problem to Solve

본 논문은 복강경 및 로봇 수술 영상에서 수술 도구 분할(Surgical Instrument Segmentation)을 수행하기 위해 필요한 고비용의 어노테이션 문제를 해결하고자 한다. 수술 도구 분할은 자율 수술 및 첨단 수술 보조 시스템을 구축하는 데 있어 핵심적인 요소이지만, 정확한 마스크를 생성하기 위해서는 고도의 전문 지식을 갖춘 의료진의 수동 작업이 필수적이므로 대규모의 레이블링된 데이터셋을 구축하는 데 막대한 비용과 시간이 소요된다.

기존의 완전 지도 학습(Fully-supervised learning) 방식은 데이터의 양에 따라 성능이 크게 좌우되지만, 수술 영상 분야에서는 자연어 이미지와 달리 대규모의 어노체이티드 데이터셋이 부족한 실정이다. 따라서 본 연구의 목표는 레이블이 없는 데이터를 효율적으로 활용하는 반지도 학습(Semi-supervised learning) 프레임워크인 SegMatch를 제안하여, 적은 양의 레이블된 데이터만으로도 높은 분할 정확도를 달성하고 모델의 강건성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 분류를 위한 반지도 학습 파이프라인인 FixMatch를 세그멘테이션 작업에 맞게 최적화하고, 학습 가능한 적대적 증강(Trainable Adversarial Augmentation) 전략을 도입하는 것이다. 주요 기여 사항은 다음과 같다.

1. **세그멘테이션 특화 변환 적용**: 분류 작업과 달리 세그멘테이션에서는 공간적 변환에 대한 등변성(Equivariance)과 광도 변환에 대한 불변성(Invariance)이 중요하다. 이를 위해 약한 증강(Weak Augmentation)에서 사용된 공간 변환을 예측 후 다시 역변환(Inverse Transformation)하여 일관성을 유지하도록 설계하였다.
2. **학습 가능한 적대적 증강 도입**: 고정된 수작업 증강(Hand-crafted augmentation)은 모델이 일정 수준 학습되면 정보 포화 상태에 이르러 더 이상 성능이 향상되지 않는 한계가 있다. 이를 해결하기 위해 I-FGSM(Iterative Fast Gradient Sign Method)을 이용한 적대적 증강을 도입하여, 학습 과정에서 동적으로 더 어려운 샘플을 생성함으로써 모델이 지속적으로 학습하도록 유도하였다.
3. **다양한 데이터셋에서의 성능 검증**: 이진 분할(Binary Segmentation) 작업(Robust-MIS 2019, EndoVis 2017)과 다중 클래스 분할(Multi-class Segmentation) 작업(CholecInstanceSeg) 모두에서 기존의 최신 반지도 학습 모델 및 완전 지도 학습 모델보다 우수한 성능을 입증하였다.

## 📎 Related Works

본 논문에서는 반지도 학습의 대표적인 기법인 Pseudo-labelling(의사 레이블링)과 Consistency Regularization(일관성 규제)을 언급하며, 이를 통합한 FixMatch의 효율성을 강조한다. 기존의 세그멘테이션 기반 반지도 학습 연구들은 픽셀 단위의 레이블 특성을 고려하여 CCT(Cross-Consistency Training)나 Cross Pseudo-Supervision 등의 방법을 제안해 왔으나, 분류 작업에서 FixMatch가 보여준 수준의 성능 향상을 세그멘테이션 분야에서 완전히 구현하지는 못했다는 점을 지적한다.

특히 수술 도구 분할 분야에서는 대부분 완전 지도 학습에 의존해 왔으며, 일부 약지도 학습(Weak supervision) 연구가 진행되었으나 실제 임상 수준의 정확도나 대규모 데이터셋에서의 일반화 성능을 입증하는 데 한계가 있었다. 또한, 적대적 학습(Adversarial Learning)이 모델의 강건성과 일반화 성능을 높인다는 점이 알려져 있음에도 불구하고, 이를 세그멘테이션을 위한 반지도 학습의 데이터 증강 전략으로 활용한 사례는 없었다는 점이 본 연구의 차별점이다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

SegMatch는 공유된 모델 파라미터 $\theta$를 사용하는 지도 학습 경로(Supervised pathway)와 비지도 학습 경로(Unsupervised pathway)로 구성된다. 지도 학습 경로는 일반적인 세그멘테이션 손실 함수를 통해 학습하며, 비지도 학습 경로는 동일한 입력 이미지에 대해 약한 증강(Weak augmentation)과 강한 증강(Strong augmentation)을 동시에 적용하는 두 개의 분기로 나뉜다.

### 주요 구성 요소 및 절차

**1. 약한 증강과 등변성 (Weak Augmentations & Equivariance)**
약한 증강 분기에서는 회전, 뒤집기, 자르기, 크기 조정과 같은 단순한 공간 변환 $\omega_e$를 적용한다. 세그멘테이션 모델은 공간 변환에 대해 등변적(Equivariant)이어야 하므로, 모델의 예측 결과에 다시 역변환 $\omega_e^{-1}$를 적용하여 원본 이미지 좌표계로 되돌린다.
$$ p^w = \omega_e^{-1}(f_\theta(\omega_e(x^u))) $$
이렇게 얻은 결과 $p^w$에 Sharpened Softmax를 적용하여 의사 레이블(Pseudo-label) $\tilde{y}$를 생성한다.

**2. 학습 가능한 강한 증강 (Trainable Strong Augmentations)**
강한 증강 분기에서는 먼저 RandAugment에서 선택된 광도 변환(Contrast, Brightness 등)을 통해 초기 이미지 $x_s^0$를 생성한다. 이후, 모델이 포화되지 않도록 I-FGSM을 이용해 적대적 섭동(Perturbation)을 추가한다. $K$번의 반복 단계 중 $k+1$번째 단계의 수식은 다음과 같다.
$$ x_s^{k+1} = \text{Clip}_{x_s^0, \epsilon} \{ x_s^k + \frac{\epsilon}{K} \cdot \text{Sign}(\nabla_{x_s^k} (L_u(f_\theta(x_s^k), \tilde{y}))) \} $$
여기서 $\epsilon$은 섭동의 최대 크기이며, $\text{Clip}$ 함수는 섭동이 $\epsilon$-근방 내에 있도록 제한한다.

**3. 손실 함수 및 학습 목표**
전체 손실 함수 $L$은 지도 학습 손실 $L_s$와 비지도 학습 손실 $L_u$의 가중 합으로 정의된다.
$$ L = L_s + w(t)L_u $$

- **지도 학습 손실 ($L_s$)**: 픽셀 단위의 교차 엔트로피 손실($l_{CE}$)과 Dice 손실($l_{DSC}$)을 결합하여 사용한다.
- **비지도 학습 손실 ($L_u$)**: 약한 증강 분기에서 생성된 의사 레이블 중 신뢰도 임계값 $t$ 이상인 픽셀들에 대해서만 강한 증강 분기의 예측 결과와 교차 엔트로피 손실을 계산한다.
- **가중치 $w(t)$**: 학습 초기에는 지도 학습에 집중하고, 시간이 흐를수록 비지도 학습의 비중을 높이기 위해 가우시안 곡선 형태의 가중치 함수를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - 이진 분할: Robust-MIS 2019, EndoVis 2017.
  - 다중 클래스 분할: CholecInstanceSeg.
- **지표**: Mean Dice, NSD (Normalized Surface Dice), IoU 기반 지표(Ch_IoU, ISI_IoU, mc_IoU).
- **비교 대상**: Mean-Teacher, WSSL, CCT, ClassMix, Min-Max Similarity, PseudoSeg, OR-Unet.

### 주요 결과

1. **반지도 학습 성능**: Robust-MIS 2019 데이터셋에서 레이블 데이터 비율이 10%와 30%일 때, SegMatch는 PseudoSeg보다 각각 2.9pp, 4.4pp 높은 Mean Dice score를 기록하며 SOTA 모델들을 압도하였다.
2. **비지도 데이터의 가치**: Robust-MIS 2019의 전체 레이블 데이터와 17K의 비지도 데이터를 함께 사용했을 때, 완전 지도 학습 모델인 OR-Unet 및 ISINet보다 각각 5.7pp, 4.8pp 높은 Dice score를 달성하였다.
3. **일반화 능력**: 학습 시 보지 못한 새로운 수술 유형(Robust-MIS 2019 Stage 3)에 대해 기존 챌린지 우승 팀보다 3.9pp 높은 성능을 보여, 비지도 데이터 활용이 모델의 일반화 성능을 크게 향상시킴을 입증하였다.
4. **다중 클래스 분할**: CholecInstanceSeg 데이터셋에서 OR-Unet 대비 mc_IoU가 25.06pp 향상되는 등 괄목할 만한 성능 개선을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 큰 성과는 적대적 증강을 통해 반지도 학습의 고질적인 문제인 '학습 포화'를 해결한 것이다. 실험 결과, 단순한 수작업 증강만 사용했을 때보다 적대적 증강을 추가했을 때 성능이 유의미하게 향상되었으며, 이는 모델이 더 어려운 샘플을 지속적으로 학습함으로써 결정 경계(Decision Boundary)를 더 정교하게 다듬었기 때문으로 해석된다. 또한, 공간 변환에 대한 역변환 적용이 세그멘테이션 작업의 특성을 정확히 반영하여 일관성 규제의 효과를 극대화하였다.

### 한계 및 비판적 해석

실패 사례 분석(Failure cases)을 통해 SegMatch가 여전히 다음과 같은 상황에서 취약함을 보였다.

- **반사 표면**: 거즈나 금속 도구의 강한 반사광이 있는 영역에서 오분류가 발생한다.
- **유사 배경**: 도구의 색상이나 질감이 주변 조직과 매우 유사한 경우 위양성(False Positive)이 나타난다.
- **분포 외 데이터 (OOD)**: 도구가 화면의 대부분을 차지하는 극단적인 케이스에서는 성능이 저하된다.

이는 비지도 데이터의 품질이 낮거나 노이즈가 심할 경우, 일관성 규제가 오히려 오류를 증폭시킬 수 있음을 시사한다. 따라서 향후 연구에서는 의사 레이블의 신뢰도를 더 정교하게 필터링하거나, 데이터 품질을 평가하는 메커니즘이 추가될 필요가 있다.

## 📌 TL;DR

SegMatch는 FixMatch 프레임워크를 세그멘테이션에 맞게 최적화하고, **I-FGSM 기반의 학습 가능한 적대적 증강**을 도입하여 수술 도구 분할 성능을 극대화한 반지도 학습 모델이다. 공간 변환의 등변성을 고려한 역변환 설계와 동적 증강 전략을 통해, 적은 양의 레이블 데이터만으로도 최신 지도 학습 모델을 능가하는 성능과 뛰어난 일반화 능력을 보여주었다. 이 연구는 레이블 확보가 어려운 의료 영상 분야에서 비지도 데이터를 활용해 임상 수준의 세그멘테이션 모델을 구축할 수 있는 실질적인 방법론을 제시하였다.
