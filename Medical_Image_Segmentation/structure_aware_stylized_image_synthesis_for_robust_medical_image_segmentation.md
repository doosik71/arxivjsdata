# STRUCTURE-AWARE STYLIZED IMAGE SYNTHESIS FOR ROBUST MEDICAL IMAGE SEGMENTATION

Jie Bao, Zhixin Zhou, Wen Jung Li, Rui Luo (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 모델이 직면한 **도메인 시프트(Domain Shift)** 문제를 해결하고자 한다. 의료 영상은 촬영 장비의 종류, 획득 조건, 그리고 환자의 개별적 특성(예: 피부색)에 따라 영상의 특성이 크게 달라지며, 이는 훈련 데이터와 테스트 데이터 간의 분포 차이를 야기하여 모델의 일반화 성능을 저하시킨다.

기존의 도메인 일반화(Domain Generalization, DG) 방법들은 훈련 세트에 테스트 도메인의 일부가 포함되어야 한다는 제약이 있어, 데이터 확보가 어려운 실제 임상 환경에서 적용하기 어렵다는 한계가 있다. 또한, 최근 주목받는 확산 모델(Diffusion Models)을 이용한 스타일 전이(Style Transfer) 방식은 이미지 생성 능력은 뛰어나지만, 정밀한 의료 분석에 필수적인 **구조적 정보(Structural Information)**를 보존하지 못하고 변형시키는 문제가 발생한다. 따라서 본 연구의 목표는 의료 영상의 해부학적 구조(병변의 위치, 크기, 모양)를 유지하면서도 서로 다른 도메인의 영상을 일관된 스타일로 변환하여, 타겟 도메인이 훈련 데이터에 없더라도 강건하고 정확한 분할 성능을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 확산 모델에 **구조 보존 네트워크(Structure-Preserving Network, SPN)**를 결합하여 구조 인식 기반의 원샷(One-shot) 이미지 스타일화를 구현하는 것이다.

단 한 장의 타겟 도메인 이미지(One-shot)만으로도 소스 도메인의 영상을 타겟 스타일로 변환할 수 있으며, 이 과정에서 SPN이 영상의 공간적 무결성을 유지함으로써 병변의 위치와 크기 변화를 최소화한다. 결과적으로 스타일 전이된 영상을 통해 모델이 도메인 간의 비선형적 차이를 극복하고, 다양한 환경에서도 일관된 분할 성능을 낼 수 있도록 유도한다.

## 📎 Related Works

### 1. 이미지 스타일 전이 (Image Style Transfer)

기존의 GAN 기반 방법(예: CycleGAN)은 비쌍으로 구성된(Unpaired) 이미지 변환이 가능했으나, 복잡한 이미지에서 세부 구조를 보존하는 데 어려움이 있었다. 이를 해결하기 위해 확산 모델(Diffusion Models)이 도입되었으며, DiffuseIT, InST, Control-Net 등이 제안되었다. 하지만 이러한 모델들조차 의료 영상과 같이 매우 정밀한 구조적 일관성이 요구되는 분야에서는 구조 보존 능력이 부족한 경우가 많다. 본 논문은 이러한 한계를 극복하기 위해 OSASIS의 SPN 구조를 채택하여 공간적 무결성을 강화하였다.

### 2. 도메인 일반화 (Domain Generalization)

도메인 일반화는 훈련 시 접하지 못한 타겟 도메인에서도 잘 작동하는 모델을 만드는 것이 목표이며, 크게 네 가지 접근 방식으로 나뉜다.

- **도메인 정렬 (Domain Alignment):** DANN, CORA 등 latent space에서 도메인 간 분포 차이를 최소화하지만, 정렬 강도가 부족할 수 있다.
- **정규화 (Regularization):** IRM, REx 등 손실 함수에 제약을 추가하여 공통 특징을 학습하지만, 계산 복잡도가 높고 훈련 시간이 길다.
- **메타 학습 (Meta-Learning):** MLDG 등 에피소드 기반 학습을 통해 빠른 적응을 꾀하지만, 태스크 설계와 파라미터 튜닝이 까다롭다.
- **데이터 증강 (Data Augmentation):** 스타일 전이를 통해 가상의 타겟 도메인 데이터를 생성하여 다양성을 높이는 방식이다.

본 논문은 데이터 증강 관점에서 스타일 전이를 활용하되, 기존 DG 방법들이 훈련 세트가 테스트 도메인을 어느 정도 대표해야 한다는 가정을 전제로 하는 것과 달리, 구조 보존 중심의 스타일 전이를 통해 훨씬 더 강건한 일반화 성능을 추구한다.

## 🛠️ Methodology

### 전체 파이프라인

제안된 방법론은 **Diffusion Model $\rightarrow$ Semantic Encoder $\rightarrow$ Structure-Preserving Network (SPN) $\rightarrow$ Segmentation Model**의 순서로 구성된다. 소스 도메인의 이미지를 타겟 도메인의 스타일로 변환한 뒤, 변환된 이미지를 분할 모델의 입력으로 사용한다.

### 주요 구성 요소 및 역할

#### 1. Diffusion Model (DDIM)

고품질의 이미지 생성을 위해 DDPM을 기반으로 하되, 샘플링 효율성과 결정론적 특성을 위해 **DDIM(Denoising Diffusion Implicit Models)**을 사용한다. 역과정(Reverse process)의 수식은 다음과 같다.
$$x_{t-1} = \sqrt{\alpha_{t-1}} f_{\theta}(x_t, t) + \sqrt{1-\alpha_{t-1}} \epsilon_{\theta}(x_t, t)$$
여기서 $f_{\theta}(x_t, t)$는 모델이 예측한 원본 이미지 $x_0$의 추정치이다.

#### 2. Semantic Encoder

이미지의 고수준 의미 정보를 캡처하여 스타일 변환 중에도 콘텐츠의 일관성을 유지하기 위해 **Diffusion Autoencoder (DiffAE)** 프레임워크를 사용한다. 인코더 $\text{Enc}_{\phi}$는 입력 이미지 $x_{in}^A$를 시맨틱 잠재 코드 $z_{sem}$으로 변환한다.
$$z_{sem} = \text{Enc}_{\phi}(x_{in}^A)$$
이 $z_{sem}$은 확산 모델의 디노이징 과정에서 조건(Condition)으로 작용하여 생성되는 이미지가 원래의 콘텐츠를 유지하도록 돕는다.

#### 3. Structure-Preserving Network (SPN)

확산 과정에서 발생할 수 있는 구조적 손실을 방지하기 위해 SPN을 도입한다. SPN은 $1 \times 1$ 컨볼루션을 통해 중간 잠재 표현의 공간 정보를 보존하며, 이를 역확산 과정의 현재 타임스텝 latent $x_t$에 직접 더해준다.
$$x_{SPN}^t = \text{SPN}(x_{in}^A)$$
$$x'_t = x_t + x_{SPN}^t$$
$$x_{t-1} = \sqrt{\alpha_{t-1}} f_{\theta}(x'_t, t, z_{in}^{sem}) + \sqrt{1-\alpha_{t-1}} \epsilon_{\theta}(x'_t, t, z_{in}^{sem})$$

#### 4. Segmentation Model

스타일 전이가 완료된 이미지는 U-Net, U-Net++, 또는 PraNet과 같은 분할 모델에 입력되어 최종 마스크를 생성한다.

### 손실 함수 (Loss Functions)

#### 스타일 전이 손실 ($\mathcal{L}_{style}$)

$\mathcal{L}_{style} = \lambda_1 \mathcal{L}_{adv} + \lambda_2 \mathcal{L}_{cycle} + \lambda_3 \mathcal{L}_{SPN}$

- **Adversarial Loss ($\mathcal{L}_{adv}$):** CLIP directional loss를 사용하여 변환된 이미지가 타겟 도메인의 스타일 특성을 갖도록 한다.
- **Cycle Consistency Loss ($\mathcal{L}_{cycle}$):** 이미지를 스타일 변환했다가 다시 원래 스타일로 되돌렸을 때 원본과 일치해야 한다는 제약을 통해 구조적 무결성을 보장한다.
- **Structure Preservation Loss ($\mathcal{L}_{SPN}$):** SPN을 통해 보존된 구조가 원본 이미지의 구조적 특징과 일치하도록 강제한다.

#### 이미지 분할 손실 ($\mathcal{L}_{seg}$)

분할 모델의 예측 마스크와 정답(Ground Truth) 간의 차이를 측정하는 일반적인 분할 손실 함수를 사용한다.
$$\mathcal{L}_{seg} = \mathbb{E}_{X_{train}, X_{label}}[\ell(f_{\theta}(X_{train}), X_{label})]$$

## 📊 Results

### 실험 설정

- **데이터셋:** 대장내시경 폴립 분할(CVC-ClinicDB $\rightarrow$ CVC-ColonDB) 및 피부 병변 분할(HAM10000) 데이터셋을 사용하였다.
- **비교 모델 (Baselines):** U-Net, U-Net++, PraNet.
- **평가 방식:** 스타일 전이를 적용하지 않은 'Direct approach'와 스타일 전이 후 학습 및 테스트를 진행한 'Style transfer approach(ST)'를 비교하였다.
- **평가 지표:** Dice, IoU, Specificity, Weighted F-measure ($F_{\beta}^w$), Structure Measure ($S_{\alpha}$), Enhanced-alignment Measure ($E_{max}^{\phi}$), MAE.

### 정량적 결과

실험 결과, 모든 베이스라인 모델에서 스타일 전이를 적용했을 때 성능이 향상되었다.

- **폴립 분할:** 특히 PraNet(ST)의 경우, Dice, IoU, $F_{\beta}^w$ 지표가 기존 PraNet 대비 약 **10% 향상**되는 결과를 보였다.
- **피부 병변 분할:** 피부톤이나 배경색의 차이가 큰 환경에서도 스타일 전이를 통해 Dice와 IoU 등의 지표가 전반적으로 약 **10% 상승**하였다.

### 정성적 결과

정성적 분석 결과, 스타일 전이를 적용한 모델이 배경 노이즈가 적고 병변의 경계를 더 정확하게 포착하는 것으로 나타났다. 또한 레이더 차트를 통해 ST 적용 시 모든 평가 지표에서 균형 있게 성능이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 확산 모델의 강력한 생성 능력과 SPN의 구조 보존 능력을 결합하여, 의료 영상 분할의 고질적인 문제인 도메인 시프트를 효과적으로 해결하였다. 특히 타겟 도메인의 데이터를 대량으로 확보할 필요 없이 단 한 장의 이미지로도 스타일을 맞출 수 있다는 점은 실제 임상 적용 가능성을 크게 높인다.

**강점 및 한계:**

- **강점:** 구조적 변형 없이 스타일만 변경함으로써 분할 모델이 학습한 해부학적 특징을 그대로 활용할 수 있게 한 점이 매우 효과적이다.
- **한계:** 현재의 파이프라인은 스타일 전이 단계와 분할 단계가 분리되어 있다. 저자들은 결론에서 스타일 전이 후의 분할 품질을 직접적으로 평가하고 최적화할 수 있는 손실 함수(loss metric)를 고안하는 것이 향후 과제임을 언급하고 있다.

**비판적 해석:**
본 연구는 스타일 전이가 분할 성능을 높인다는 것을 입증했지만, 스타일 전이 모델 자체가 매우 무겁기 때문에 실시간 진단 시스템에 적용하기 위해서는 추론 속도 최적화에 대한 논의가 추가적으로 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 의료 영상의 도메인 시프트를 해결하기 위해 **구조 보존 네트워크(SPN)**가 결합된 **확산 모델 기반의 원샷 스타일 전이** 방법을 제안하였다. 이 방법은 병변의 해부학적 구조를 유지하면서 영상의 스타일만 타겟 도메인에 맞게 변환하여, U-Net, PraNet 등 기존 분할 모델의 강건성과 정확도를 약 10%가량 향상시켰다. 이는 타겟 도메인 데이터가 부족한 환경에서도 높은 성능의 의료 영상 분할을 가능케 하여 향후 정밀 진단 시스템의 신뢰성을 높이는 데 기여할 것으로 기대된다.
