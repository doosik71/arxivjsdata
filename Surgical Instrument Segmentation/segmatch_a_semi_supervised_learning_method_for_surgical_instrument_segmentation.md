# SegMatch: Semi-supervised surgical instrument segmentation

Meng Wei, Charlie Budd, Luis C. Garcia-Peraza-Herrera, Reuben Dorent, Miaojing Shi, and Tom Vercauteren (2023)

## 🧩 Problem to Solve

수술 도구 분할(Surgical instrument segmentation)은 첨단 수술 보조 및 컴퓨터 보조 중재술(computer-assisted interventions)을 가능하게 하는 핵심 기술이다. 하지만 딥러닝 기반의 분할 모델을 학습시키기 위해서는 정밀한 픽셀 단위의 주석(annotation)이 포함된 대규모 데이터셋이 필요하다.

수술 영상의 주석 작업은 높은 전문성을 요구하므로 비용이 많이 들고 시간이 오래 걸린다는 치명적인 단점이 있다. 이로 인해 일반 자연 이미지 데이터셋과 달리 수술 도구 분할 분야에서는 대규모의 정밀 주석 데이터셋을 확보하기가 매우 어렵다. 결과적으로 모델의 강건성(robustness)과 정밀도를 확보하는 데 큰 장벽이 되며, 이는 실제 임상 현장에 배포하는 것을 어렵게 만든다.

본 논문의 목표는 라벨이 없는(unlabelled) 데이터를 효과적으로 활용하는 준지도 학습(Semi-supervised Learning, SSL) 방법론인 **SegMatch**를 제안하여, 값비싼 주석 데이터에 대한 의존도를 낮추면서도 높은 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 분류를 위한 SSL 파이프라인인 FixMatch를 분할 작업(segmentation task)에 맞게 최적화하여 적용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **분할 작업에 특화된 등변성(Equivariance) 및 불변성(Invariance) 고려**: 분류 모델은 모든 증강에 대해 불변성(invariant)을 가져야 하지만, 분할 모델은 광도 변화(photometric)에는 불변해야 하고 공간적 변화(spatial)에는 등변(equivariant)해야 한다. 이를 해결하기 위해 약한 증강(weak augmentation) 단계에서 적용한 공간 변환을 예측 후 다시 역변환(inverse transformation)하여 일관성을 맞추는 구조를 도입하였다.
2. **학습 가능한 적대적 증강(Trainable Adversarial Augmentation) 전략**: 고정된 수작업 증강(hand-crafted augmentation)은 모델이 일정 수준 학습되면 정보 포화 상태에 이르러 더 이상 성능이 향상되지 않는 문제가 있다. 이를 극복하기 위해 I-FGSM(Iterative Fast Gradient Sign Method)을 이용한 적대적 공격 기반의 증강 방식을 도입하여, 학습 과정에서 동적으로 더 어려운 샘플을 생성함으로써 모델의 일반화 성능을 극대화하였다.

## 📎 Related Works

### 준지도 학습 (Semi-supervised Learning)

- **Pseudo-labelling 및 Consistency Regularization**: 라벨이 없는 데이터에 대해 모델이 예측한 값을 정답으로 사용하거나, 동일한 데이터에 다른 변형을 가했을 때 예측값이 일관되게 유지되도록 유도하는 방식이다. FixMatch는 이 두 가지를 결합하여 분류 작업에서 매우 높은 성능을 보였다.
- **분할을 위한 SSL**: CCT(Cross-Consistency Training)나 Cross Pseudo-Supervision 등이 제안되었으나, 분류 모델 기반의 SSL만큼의 성능 향상을 보이지 못했다. 저자들은 그 이유가 변환에 대한 등변성/불변성 처리 부족과 현대적인 증강 전략의 효율적 활용 부재에 있다고 분석한다.

### 수술 도구 분할 (Surgical Instrument Segmentation)

- 대부분의 연구가 지도 학습(fully-supervised)에 의존하며, OR-Unet과 같은 최적화된 U-Net 구조가 널리 사용된다.
- 약한 지도 학습(weak supervision)이나 도메인 적응(domain adaptation) 연구가 일부 존재하지만, FixMatch 스타일의 SSL을 수술 도구 분할에 본격적으로 적용한 사례는 부족한 실정이다.

### 적대적 학습 (Adversarial Learning)

- 적대적 샘플은 모델을 속이기 위해 설계된 미세한 섭동(perturbation)이 추가된 이미지이다. 적대적 공격에 강건한 모델은 일반적으로 일반화 능력이 더 뛰어난 경향이 있으며, 본 논문은 이를 SSL의 증강 전략으로 활용하였다.

## 🛠️ Methodology

### 전체 시스템 구조

SegMatch는 공유된 파라미터 $\theta$를 가진 두 개의 경로(Path)로 구성된다.

1. **지도 학습 경로 (Supervised Pathway)**: 라벨이 있는 데이터($D_l$)를 사용하여 표준적인 분할 손실 함수로 학습한다.
2. **준지도 학습 경로 (Unsupervised Pathway)**: 라벨이 없는 데이터($D_u$)를 사용하여 **약한 증강 브랜치**와 **강한 증강 브랜치** 사이의 일관성을 학습한다.

### 약한 증강, 등변성 및 의사 라벨 생성

약한 증강 브랜치에서는 회전, 뒤집기, 자르기 등의 공간적 변환 $\omega_e$를 적용한다. 분할 모델은 공간 변환에 대해 등변(equivariant)해야 하므로, 모델의 예측 결과에 다시 역변환 $\omega_e^{-1}$를 적용하여 원래 이미지 좌표계로 되돌린다.

이렇게 얻은 결과 $p^w = \omega_e^{-1}(f_\theta(\omega_e(x^u)))$에 대해, 온도 하이퍼파라미터 $T$를 이용한 Sharpened Softmax를 적용하여 확신도가 높은 픽셀에 대해서만 의사 라벨(pseudo-label) $\tilde{y}$를 생성한다.

$$\tilde{y}_i = \text{Sharpen}(\text{Softmax}(p^w_i), T)$$

### 학습 가능한 강한 증강 (Trainable Strong Augmentations)

강한 증강 브랜치는 두 단계로 구성된다.

1. **초기화**: RandAugment에서 선택된 광도 변환(대비, 밝기, 색상 등) 3개를 무작위로 조합하여 초기 강한 증강 이미지 $x_0^s$를 생성한다.
2. **적대적 섭동 (I-FGSM)**: 고정된 증강의 한계를 넘기 위해, 의사 라벨 $\tilde{y}$와 모델 예측 간의 손실을 최대화하는 방향으로 이미지를 미세하게 변형시킨다. $K$번의 반복 단계 동안 다음과 같이 업데이트한다.

$$x_{k+1}^s = \text{Clip}_{x_0^s, \epsilon} \{ x_k^s + \frac{\epsilon}{K} \cdot \text{Sign}(\nabla_{x_k^s} (L_u(f_\theta(x_k^s), \tilde{y}))) \}$$

여기서 $\epsilon$은 섭동의 최대 크기를 제한하는 하이퍼파라미터이며, 이를 통해 이미지의 본질적인 특성을 유지하면서도 모델이 학습하기 어려운 샘플을 생성한다.

### 손실 함수 (Loss Functions)

전체 손실 함수는 지도 학습 손실 $L_s$와 준지도 학습 손실 $L_u$의 가중합으로 정의된다.

1. **지도 학습 손실 ($L_s$)**: 픽셀 단위 교차 엔트로피(Cross-Entropy) 손실과 Dice 손실의 합이다.
    $$L_s = \frac{1}{|D_l|} \sum_{x^l \in D_l} (l_{DSC}(y^l, f_\theta(x^l)) + l_{CE}(y^l, f_\theta(x^l)))$$
2. **준지도 학습 손실 ($L_u$)**: 확신도 임계값 $t$를 넘는 픽셀들에 대해서만 의사 라벨 $\tilde{y}$와 강한 증강 브랜치의 예측 $p$ 사이의 교차 엔트로피를 계산한다.
    $$L_u = \frac{1}{|D_u|} \sum_{x^u \in D_u} \frac{1}{|N_{x^u}^v|} \sum_{i \in N_{x^u}^v} l_{CE}(\tilde{y}_i, p_i)$$
3. **최종 손실**: $L = L_s + w(t)L_u$ (단, $w(t)$는 학습 초기에는 $L_s$에 집중하고 점차 $L_u$의 비중을 높이는 Gaussian ramp-up 함수이다.)

## 📊 Results

### 실험 설정

- **데이터셋**: Robust-MIS 2019(복강경), EndoVis 2017(로봇 수술), CholecInstanceSeg(다중 클래스)
- **모델 및 지표**: OR-Unet을 백본으로 사용하였으며, Binary task에서는 Mean Dice와 NSD(Normalized Surface Dice)를, Multi-class task에서는 IoU 기반 지표($Ch\_IoU, ISI\_IoU, mc\_IoU$)를 사용하였다.
- **비교 대상**: Mean-Teacher, WSSL, CCT, ClassMix, PseudoSeg 등 SOTA SSL 모델 및 지도 학습 모델(OR-Unet, ISINet)

### 주요 결과

1. **SSL 모델 간 비교**: Robust-MIS 2019와 EndoVis 2017 데이터셋 모두에서 SegMatch가 다른 SSL 모델들을 유의미하게 앞섰다. 특히 PseudoSeg 대비 Robust-MIS 2019에서 최대 4.4 pp(percentage points)의 Dice score 향상을 보였다.
2. **지도 학습 모델과의 비교**: 동일한 양의 라벨 데이터를 사용했을 때, 라벨이 없는 데이터를 추가로 활용한 SegMatch가 지도 학습 전용 OR-Unet보다 일관되게 높은 성능을 기록하였다. 라벨 데이터 비율이 낮을수록(예: 10%) SSL의 이점이 더 크게 나타났다.
3. **일반화 능력**: 학습 단계에서 보지 못한 수술 유형(Robust-MIS 2019의 Stage 3)에서도 SegMatch가 기존 챌린지 우승팀보다 3.9 pp 높은 성능을 보이며 매우 뛰어난 일반화 성능을 입증하였다.
4. **다중 클래스 분할**: CholecInstanceSeg 데이터셋에서 라벨 데이터 없이 추가 66.1k의 무라벨 데이터를 활용함으로써 $Ch\_IoU$ 기준 OR-Unet 대비 24.2 pp라는 압도적인 성능 향상을 달성하였다.

### 절제 연구 (Ablation Study)

- **적대적 증강의 효과**: I-FGSM을 제거하고 수작업 증강만 사용했을 때 성능이 하락하였다. 이는 적대적 증강이 정보 포화를 방지하고 모델이 지속적으로 학습하게 함을 의미한다.
- **증강 강도의 영향**: $\epsilon = 0.08$일 때 최적의 성능을 보였으며, 너무 큰 $\epsilon$ 값은 오히려 성능 저하를 일으켰다.
- **일관성 규제 vs 의사 라벨링**: 강한 증강을 제거했을 때의 성능 하락이 약한 증강을 제거했을 때보다 컸으며, 이는 본 파이프라인에서 Pseudo-labelling보다 Consistency Regularization이 더 핵심적인 역할을 수행함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 성과

SegMatch는 단순히 분류 모델을 가져온 것이 아니라, 분할 작업의 핵심인 **등변성(Equivariance)**을 역변환을 통해 해결하고, **적대적 학습**을 통해 데이터 증강의 한계를 극복하였다. 특히 수술 도구라는 특수 도메인에서 라벨링 비용 문제를 해결하면서도, 보지 못한 수술 환경에 대한 강건한 일반화 능력을 보여준 점이 매우 고무적이다.

### 한계 및 비판적 해석

실패 사례 분석(Failure cases)을 통해 다음과 같은 한계가 드러났다.

- **반사 표면 및 유사 배경**: 거즈와 같은 반사성 표면이나 배경 조직과 색상이 매우 유사한 도구의 경우 의사 라벨 자체가 부정확하게 생성되어 성능이 저하된다.
- **노이즈 증폭**: 일관성 규제(Consistency regularization) 특성상 무라벨 데이터에 노이즈가 많거나 이미지 품질이 낮으면 오류가 증폭될 위험이 있다.
- **분포 외(OOD) 데이터**: 도구가 화면을 거의 다 채우는 극단적인 이미지 등 분포를 벗어난 샘플에 대해서는 여전히 취약한 모습을 보인다.

## 📌 TL;DR

본 논문은 수술 도구 분할을 위한 준지도 학습 프레임워크인 **SegMatch**를 제안한다. FixMatch 구조를 기반으로 하되, **(1) 공간 변환에 대한 역변환 적용으로 등변성을 확보**하고, **(2) I-FGSM 기반의 학습 가능한 적대적 증강을 통해 학습 포화를 방지**하였다. 실험 결과, 적은 양의 라벨 데이터만으로도 기존 SOTA 모델 및 지도 학습 모델을 능가하는 성능을 보였으며, 특히 처음 보는 수술 환경에 대한 일반화 능력이 뛰어나 임상 현장의 데이터 부족 문제를 해결할 강력한 대안이 될 가능성이 높다.
