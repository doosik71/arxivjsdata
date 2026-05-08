# Leveraging Labelled Data Knowledge: A Cooperative Rectification Learning Network for Semi-supervised 3D Medical Image Segmentation

Yanyan Wang, Kechen Song, Yuyuan Liu, Shuai Ma, Yunhui Yan, Gustavo Carneiro (2025)

## 🧩 Problem to Solve

본 논문은 소량의 라벨링된 데이터와 대량의 라벨링되지 않은 데이터를 활용하는 Semi-supervised 3D medical image segmentation 문제에 집중한다. 3D 의료 영상의 경우 픽셀 단위의 정밀한 어노테이션을 획득하는 데 막대한 비용과 시간이 소모되므로, 라벨이 없는 데이터를 효과적으로 학습에 활용하는 것이 매우 중요하다.

기존의 반지도 학습 방법들은 주로 Teacher-Student 구조의 Consistency Learning 전략을 사용한다. 그러나 Teacher 네트워크가 생성한 Pseudo-labels에 오류가 포함되어 있을 경우, Student 네트워크가 이 잘못된 정보를 학습하여 오류를 더욱 강화하는 Confirmation Bias 현상이 발생한다. 이를 해결하기 위해 기존 연구들은 주로 예측의 불확실성(Uncertainty)을 측정하여 신뢰도가 낮은 영역을 필터링하는 방식을 사용했다. 하지만 이러한 방식은 학습 초기 단계에서 너무 많은 유용한 데이터를 버리게 되어 학습 효율을 떨어뜨리며, 신뢰도가 높다고 판단된 영역조차 실제로는 잘못된 예측일 수 있다는 한계가 있다. 따라서 본 논문의 목표는 라벨링된 데이터의 지식을 활용하여 Pseudo-labels를 능동적으로 교정(Rectification)함으로써, Confirmation Bias를 줄이고 더 많은 비라벨 데이터를 학습에 활용하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 라벨링된 데이터로부터 클래스별 대표 특성인 Prototypes를 학습하고, 이를 외부 지식 사전(Knowledge Priors)으로 사용하여 Pseudo-labels를 픽셀 수준에서 교정하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Cooperative Rectification Learning Network (CRLN):** 각 클래스당 여러 개의 Prototype을 학습하여 이를 통해 Pseudo-labels를 적응적으로 교정하는 네트워크를 제안한다.
2. **Dynamic Interaction Module (DIM):** Prototype과 다중 해상도 이미지 특징 간의 쌍별(Pairwise) 및 클래스 간(Cross-class) 상호작용을 촉진하여, Pseudo-label 교정을 위한 정밀한 픽셀 수준의 단서(Clues)를 생성한다.
3. **Collaborative Positive Supervision (CPS):** 클래스 경계와 같이 구분이 어려운 불확실한 영역(Uncertain regions)의 변별력을 높이기 위해, 학습된 Prototype과 클래스 평균 표현을 결합한 'Unassertive positive' 샘플을 정의하고 대조 학습(Contrastive Learning)을 수행하는 메커니즘을 제안한다.

## 📎 Related Works

의료 영상 분할 분야에서는 UNet과 VNet과 같은 Encoder-Decoder 구조가 표준적으로 사용되어 왔으며, 최근에는 Vision Transformer(ViT)나 Diffusion Probabilistic Model(DPM)을 결합한 방식들이 등장하고 있다. 반지도 학습(Semi-supervised Learning) 관점에서는 Mean Teacher와 같은 Consistency Regularization 전략이 널리 쓰인다.

기존의 Confirmation Bias 해결 방식은 크게 두 가지이다. 첫째는 여러 개의 Teacher 모델을 사용하여 다양한 관점의 Pseudo-labels를 생성하는 방식이지만, 각 모델의 예측이 일관되지 않아 항상 성능 향상을 보장하지 않는다. 둘째는 불확실성 기반 필터링 방식(예: UA-MT)인데, 이는 국소적인 픽셀 정보에만 의존하여 전역적인 맥락을 놓치며, 학습 초기 단계에서 데이터 활용도를 극도로 낮춘다는 단점이 있다. 본 논문은 이러한 한계를 극복하기 위해 라벨링된 데이터의 전역적 지식(Prototypes)을 명시적으로 활용하는 교정 방식을 제안하여 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 파이프라인

본 모델은 기본적으로 Teacher-Student 구조를 따르며, Student 모델은 강한 증강(Strong Augmentation)된 데이터를, Teacher 모델은 약한 증강(Weak Augmentation)된 데이터를 입력받아 Pseudo-label을 생성한다. 전체 프로세스는 크게 CRLN의 **학습 단계(Learning Stage)**와 **교정 단계(Rectification Stage)**, 그리고 **CPS 메커니즘**으로 구성된다.

### 2. Cooperative Rectification Learning Network (CRLN) 및 DIM

CRLN은 라벨링된 데이터에서 클래스별 Prototype $P \in \mathbb{R}^{C \times R \times F}$를 학습한다. 여기서 $C$는 클래스 수, $R$은 클래스당 Prototype의 수, $F$는 특징 공간의 차원이다.

**Dynamic Interaction Module (DIM)**은 이 Prototype들을 정밀하게 다듬고 교정 맵을 생성하는 역할을 한다.

- **학습 과정:** Prototype들을 $R$개의 행렬로 나누고, Student Decoder의 중간 레이어(2번째, 3번째 레이어)에서 추출된 특징들과 Cross-attention을 수행한다. 이를 통해 Prototype들이 텍스처와 시맨틱 정보를 점진적으로 흡수하도록 한다.
- **교정 맵 생성:** 생성된 근접도 행렬(Proximity matrices)에 $3 \times 3 \times 3$ 컨볼루션을 적용하여 공간적 일관성을 부여하고, $1 \times 1 \times 1$ 컨볼루션을 통해 클래스 간 상호작용을 반영한 전역 관계 맵 $M(x) \in \mathbb{R}^{C \times H \times W \times D}$를 생성한다.
- **Pseudo-label 교정:** 최종적으로 Teacher가 생성한 Pseudo-label $\bar{y}$를 다음과 같은 방정식으로 교정한다.
$$\bar{y}_r = \bar{y} + (1 - \mu) \times M(x)$$
여기서 $\mu \in [0, 1]$는 교정 정도를 조절하는 학습 가능한 파라미터이다. 학습 초기에는 $\mu$ 값이 낮아 외부 지식($M(x)$)의 영향력이 크고, 모델이 성숙해짐에 따라 $\mu$가 증가하여 자체 예측값의 비중이 높아진다.

### 3. Collaborative Positive Supervision (CPS)

불확실한 영역(예: 클래스 경계)의 변별력을 높이기 위해 InfoNCE Loss 기반의 대조 학습을 수행한다.

- **Anchor Set:** 예측 신뢰도가 낮은 불확실한 영역의 표현(Representation)들을 샘플링한다.
- **Negative Set:** 해당 클래스가 아닌 영역의 표현들을 구성한다.
- **Positive Set (Unassertive Positive):** 단순히 클래스 평균을 사용하는 대신, 학습된 Prototype의 평균과 현재 클래스의 평균 표현을 $\xi$ 비율로 결합하여 생성한다.
$$\text{Positive Sample} = \frac{r_m + \xi \times \text{mean}(P^c)}{1 + \xi}, \quad \xi \sim U(0, 1)$$
이를 통해 너무 극단적이지 않으면서도 클래스의 핵심 특징을 가진 '적당한' 정답 샘플을 제공함으로써, 모델이 불확실한 영역을 더 잘 구분하게 만든다.

### 4. 전체 학습 목표 (Overall Loss)

최종 손실 함수는 다음과 같이 정의된다.
$$\ell(L, U, \theta, P) = \frac{1}{|L|} \sum_{(x,y) \in L} (\ell_s(f_{\theta}(x), y) + \ell_s(M(x), y)) + \frac{1}{|U|} \sum_{x \in U} \ell_u(f_{\theta}(A_s(x)), \bar{y}_r) + \ell_{cp}(L, U, \theta)$$
여기서 $\ell_s$는 지도 학습 손실, $\ell_u$는 교정된 Pseudo-label $\bar{y}_r$를 이용한 비지도 학습 손실, $\ell_{cp}$는 CPS 대조 학습 손실이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Left Atrium (LA), Pancreas-CT, BraTS19의 세 가지 3D 의료 영상 데이터셋을 사용하였다.
- **백본 네트워크:** LA와 Pancreas-CT는 VNet을, BraTS19는 3D-UNet을 사용하였다.
- **평가 지표:** Dice, Jaccard, ASD(Average Surface Distance), 95HD(95% Hausdorff Distance)를 사용하였다.
- **비교 대상:** UA-MT, TraCoCo, RCPS 등 19개의 최신 반지도 학습 방법론과 비교하였다.

### 2. 주요 결과

- **정량적 성능:** 세 데이터셋 모두에서 기존 SOTA 방법론들을 뛰어넘는 성능을 기록하였다. 특히 LA 데이터셋에서 8개의 라벨 샘플만 사용했을 때, Jaccard 지수와 95HD에서 기존 SOTA인 RCPS를 각각 1.69%와 3.31(Voxel)만큼 개선하였다.
- **학습 효율:** Figure 7에 따르면, 제안 방법은 CAC4SSL, TraCoCo 등보다 학습 시간이 훨씬 짧으며, 기존 Teacher-Student 기반 모델들과 유사한 수준의 효율성을 보이면서도 성능은 훨씬 높다.
- **SAM 모델 비교:** Auto-SAM, MA-SAM과 같은 거대 모델 기반 방법론과 비교했을 때도 95HD 지표 등에서 압도적인 우위를 보였다.

### 3. 절제 연구 (Ablation Study)

- **CRLN+DIM의 효과:** Pseudo-labels의 신뢰도(Reliable predictions) 비율이 교정 후 유의미하게 상승하였으며, 이는 더 많은 비라벨 데이터가 학습에 기여하게 함으로써 성능 향상으로 이어짐을 확인하였다.
- **CPS의 효과:** 경계 영역에서의 예측 자신감(Confidence map)이 향상되었으며, 특히 Prototype과 평균 표현을 결합한 전략이 단독 사용 시보다 효과적임을 입증하였다.
- **Prototype 수:** 클래스당 1개의 Prototype보다 여러 개(최적 16개)를 사용하는 것이 복잡한 의료 영상의 특성을 더 잘 포착함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Pseudo-labels의 품질 문제를 단순한 '필터링'이 아닌 '교정'의 관점에서 접근하여 성공적인 결과를 얻었다. 특히 라벨링된 데이터에서 추출한 Prototype을 외부 지식으로 활용함으로써, 비지도 학습의 고질적인 문제인 Confirmation Bias를 효과적으로 억제하였다.

**강점:**

- 라벨링된 데이터의 지식을 전역적으로 활용하여 Pseudo-label의 질을 직접적으로 높였다.
- 단순한 평균값이 아닌 다중 Prototype과 동적 상호작용(DIM)을 통해 의료 영상의 복잡한 변동성을 수용하였다.
- 불확실한 영역에 대해 'Unassertive'라는 완화된 정답지를 제공하는 CPS 전략을 통해 학습의 안정성을 높였다.

**한계 및 비판적 해석:**

- **세밀한 구조 분할의 어려움:** 폐정맥(Pulmonary veins)과 같이 매우 작고 픽셀 수가 적은 영역에서는 여전히 분할 성능이 떨어진다. 이는 Prototype 기반의 전역적 지식이 국소적인 미세 구조를 포착하기에는 부족할 수 있음을 시사한다.
- **과도한 평활화(Over-smoothing):** 췌장 표면의 자연스러운 굴곡을 제대로 캡처하지 못하고 결과물이 너무 매끄럽게 나오는 경향이 있다. 이는 모델이 고주파 세부 정보보다는 저주파 전역 특징에 치중하여 학습되었을 가능성이 크다.
- **임상적 적용 검증 부재:** 기술적인 지표 향상은 뚜렷하지만, 이것이 실제 임상 진단이나 치료 계획 수립에 어떤 실질적인 이득을 주는지에 대한 분석은 포함되지 않았다.

## 📌 TL;DR

본 논문은 3D 의료 영상 반지도 분할에서 Pseudo-labels의 오류로 인한 Confirmation Bias를 해결하기 위해, 라벨링된 데이터로부터 학습한 **다중 Prototype**을 활용하는 **Cooperative Rectification Learning Network (CRLN)**를 제안한다. **DIM** 모듈을 통해 픽셀 수준의 정밀한 교정 맵을 생성하고, **CPS** 메커니즘으로 경계 영역의 변별력을 높임으로써 LA, Pancreas, BraTS 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 단순한 데이터 필터링을 넘어 라벨링된 지식을 이용해 비라벨 데이터를 능동적으로 정제하는 방향을 제시하며, 향후 미세 구조 보존을 위한 세부 손실 함수 설계 등의 연구로 확장될 가능성이 높다.
