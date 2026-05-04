# Analysing Diffusion Segmentation for Medical Images

Mathias Ottl, Siyuan Mei, Frauke Wilm, Jana Steenpass, Matthias Rübner, Arndt Hartmann, Matthias Beckmann, Peter Fasching, Andreas Maier, Ramona Erber, and Katharina Breininger (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 최근 주목받고 있는 Diffusion 모델의 적용 방식을 비판적으로 분석하는 것을 목표로 한다. Denoising Diffusion Probabilistic Models(DDPM)는 확률적 모델링과 다양한 결과물 생성이 가능하다는 점 때문에 이미지 생성뿐만 아니라 분할 작업에도 도입되었다. 특히 의료 영상 분야에서는 레이블 노이즈가 흔하고 예측의 불확실성(Uncertainty)을 측정하는 것이 매우 중요하기 때문에, Diffusion 모델을 통한 불확실성 캡처 능력이 매력적인 요소로 작용한다.

하지만 기존 연구들은 성능 향상을 위한 강력한 아키텍처 제안에만 집중했을 뿐, 다음의 핵심적인 질문들에 대한 답을 제시하지 않았다. 첫째, Diffusion 기반의 분할 작업이 일반적인 이미지 생성 과정과 어떻게 다른지, 특히 학습 동작(Training behavior) 측면에서 어떤 차이가 있는지가 명확히 분석되지 않았다. 둘째, 제안된 새로운 아키텍처들이 제공하는 성능 향상이 Diffusion 과정 자체에서 오는 이점인지, 아니면 단순히 아키텍처의 개선으로 인한 일반적인 분할 성능의 향상인지에 대한 구분이 부족했다. 따라서 본 연구는 Diffusion 분할의 특성을 심층 분석하여 향후 더 나은 설계와 평가 방법을 제공하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Diffusion 분할 모델의 동작 원리를 이론적, 실험적으로 분석하여 이미지 생성 모델과의 차이점을 규명한 것이다. 주요 기여 사항은 다음과 같다.

1. **학습 방식에 따른 성능 및 불확실성 비교**: 동일한 아키텍처를 사용하여 Feed-forward 방식의 학습과 Diffusion 방식의 학습 결과를 비교함으로써, Diffusion 모델이 제공하는 특유의 이점(특히 불확실성 정량화 및 Calibration)을 확인하였다.
2. **분할과 생성의 학습 동작 차이 분석**: 이미지 생성 시의 손실 함수 동작과 분할 시의 동작이 서로 다름을 밝혀냈으며, 기존의 이미지 생성용 Diffusion 스케줄이 분할 작업에는 부적합할 수 있다는 인사이트를 제시하였다.
3. **데이터셋 특성에 따른 Diffusion 동작 분석(Dataset Fingerprints)**: 의료 영상의 특성(객체 크기, 분포 등)에 따라 Diffusion 과정에서의 정보 손실 속도가 다름을 분석하고, 이에 따라 Diffusion 프로세스를 데이터셋별로 최적화해야 할 필요성을 제안하였다.

## 📎 Related Works

논문에서는 Diffusion 기반 분할을 위해 제안된 세 가지 주요 아키텍처를 언급한다.
- **EnsemDiff**: UNet 구조를 기반으로 하며, 조건이 되는 이미지(Conditioning image)를 노이즈 입력에 단순 결합(Concatenate)하는 방식을 사용한다.
- **SegDiff**: 별도의 인코더를 통해 조건 이미지의 임베딩을 추출하고, 이를 각 해상도 단계의 인코더 특징(Feature)에 더해주는 방식을 취한다.
- **MedSegDiff**: SegDiff와 유사하게 인코더를 사용하지만, 중간 해상도 단계에서 Feature Frequency Parsers를 통해 특징을 병합하는 더 복잡한 구조를 가진다.

기존 연구들은 이러한 아키텍처들이 분할 성능을 높였다고 주장하지만, 본 논문은 이러한 성능 향상이 Diffusion 모델의 특성 덕분인지 아니면 단순히 네트워크 구조의 개선 때문인지에 대한 분석이 결여되어 있었다는 점을 한계로 지적하며 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 학습 목표
Diffusion 모델은 데이터를 점진적으로 파괴하는 Forward process와 이를 다시 복원하는 Reverse process로 구성된다. 본 논문에서는 세 가지 학습 목표 함수를 정의한다.

첫째, 일반적인 노이즈 예측 목표 함수 $\mathcal{L}_{DM}$은 다음과 같다.
$$\mathcal{L}_{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} [\|\epsilon - \epsilon_\theta(x_t, t)\|^2_2]$$
여기서 $\epsilon$은 추가된 노이즈이며, $\epsilon_\theta$는 모델이 예측한 노이즈이다.

둘째, 노이즈 대신 원본 데이터 $x_0$를 직접 예측하는 목표 함수 $\mathcal{L}'_{DM}$이다.
$$\mathcal{L}'_{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} [\|x_0 - \epsilon_\theta(x_t, t)\|^2_2]$$

셋째, 분할 작업과 같이 조건 $y$(원본 이미지)가 주어진 경우의 Conditional Diffusion 목표 함수 $\mathcal{L}_{CDM}$이다.
$$\mathcal{L}_{CDM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} [\|\epsilon - \epsilon_\theta(x_t, t, y)\|^2_2]$$
여기서 $x$는 분할 마스크(Segmentation mask)이며, $y$는 분할 대상이 되는 조건 이미지이다.

### 실험 설계 (Four Experiment Sets)
연구진은 분석을 위해 네 가지 실험 세트를 구성하였다.
- **E1 (Feed-forward Segmentation)**: Diffusion 과정을 제외하고 Dice loss와 Cross-entropy loss를 사용하여 고전적인 방식으로 학습한다. Diffusion 아키텍처의 순수 구조적 성능을 측정하기 위함이다.
- **E2 (Diffusion Segmentation)**: $\mathcal{L}_{CDM}$을 사용하여 $\epsilon$을 예측하도록 설계된 원래의 Diffusion 학습 방식이다.
- **E3 (Mask Prediction)**: 이미지 입력 없이 노이즈 섞인 마스크 $x_t$로부터 원본 마스크 $x_0$를 복원하는 $\mathcal{L}'_{DM}$ 방식을 학습하여, 마스크 자체의 형태 생성 능력을 분석한다.
- **E4 (Image Generation)**: 마스크가 아닌 일반 이미지 $x$를 대상으로 $\mathcal{L}_{DM}$을 학습하여 이미지 생성과 분할의 차이를 비교한다.

## 📊 Results

### 성능 및 불확실성 평가
ISIC16(피부 병변), MoNuSeg(세포핵), HER2(유방암 조직) 데이터셋을 사용하여 IoU(Intersection over Union)와 ECE(Expected Calibration Error)를 측정하였다.

- **정량적 결과**: 동일한 아키텍처 내에서는 Feed-forward 방식보다 Diffusion 방식으로 학습했을 때 IoU와 ECE 모두 개선되는 경향을 보였다. 특히 ECE 수치가 낮아진 것은 Diffusion 모델이 모델의 불확실성을 더 잘 정량화(Calibration)하고 있음을 시사한다.
- **SOTA와의 비교**: 하지만 Diffusion 기반 모델들의 절대적인 성능은 nnU-Net이나 SegFormer와 같은 최신 Feed-forward 전용 모델들보다 낮게 나타났다. 이는 Diffusion 방식이 아키텍처 자체의 성능보다는 확률적 모델링과 불확실성 표현에 이점이 있음을 보여준다.

### 학습 동작 및 데이터셋 분석
- **Loss 동작**: 이미지 생성(E4)의 손실 함수는 $t$가 증가함에 따라 단조 감소하는 특성을 보이지만, Diffusion 분할(E2)은 초기 단계($t < 50$)에서 급격히 낮아졌다가 다시 상승하는 비단조적(Non-monotonic)인 거동을 보인다. 이는 마스크의 넓은 상수 영역 때문에 초기 단계에서는 단순한 고주파 필터만으로도 노이즈 예측이 쉽기 때문으로 분석된다.
- **데이터셋 핑거프린트(Fingerprints)**: 데이터셋마다 $x_t$에서 $x_0$를 복원하는 오차(MSE)의 변화 양상이 달랐다.
    - **MoNuSeg**: 작은 객체가 많아 노이즈가 조금만 섞여도 빠르게 정보를 상실한다.
    - **ISIC16**: 객체가 크고 중심에 위치하여 높은 $t$ 값에서도 형태가 비교적 잘 유지된다.

## 🧠 Insights & Discussion

본 논문은 Diffusion 분할 모델이 단순히 생성 모델의 구조를 차용한 것이 아니라, 분할 마스크라는 데이터의 특수성을 고려한 설계가 필요함을 역설한다.

**강점 및 발견**:
- Diffusion 학습이 동일 구조의 Feed-forward 학습보다 Calibration 성능(ECE)을 높인다는 점을 입증하였다.
- 분할 마스크는 이미지와 달리 세부 디테일이 적어 정보 손실이 연속적이지 않으며, 이로 인해 기존 이미지 생성용 Noise schedule이 비효율적일 수 있음을 밝혀냈다.

**한계 및 비판적 해석**:
- Diffusion 기반 분할 모델들이 여전히 nnU-Net 같은 SOTA 모델에 비해 낮은 IoU를 기록한다는 점은, 현재의 Diffusion 접근 방식이 단순히 "불확실성 측정"이라는 부가 가치를 위해 성능 희생을 감수하고 있는 상태임을 의미한다.
- 논문에서 제안한 "데이터셋별 최적화 스케줄"은 가설 단계이며, 실제로 이를 적용하여 성능을 얼마나 끌어올릴 수 있는지에 대한 정량적 검증은 본문에서 충분히 다루어지지 않았다.

## 📌 TL;DR

이 논문은 의료 영상 분할에 적용된 Diffusion 모델이 일반적인 이미지 생성 모델과 학습 동작 및 정보 손실 과정에서 근본적인 차이가 있음을 분석하였다. 특히 동일 아키텍처 대비 Diffusion 학습이 불확실성 측정 성능(ECE)을 향상시키지만, 절대적 성능은 여전히 SOTA Feed-forward 모델에 뒤처짐을 확인하였다. 또한 데이터셋의 특성(객체 크기 등)에 따라 Diffusion 프로세스를 다르게 설계해야 한다는 '데이터셋 핑거프린트' 개념을 제시하여, 향후 의료 영상 분할을 위한 맞춤형 Diffusion 모델 설계의 방향성을 제시하였다.