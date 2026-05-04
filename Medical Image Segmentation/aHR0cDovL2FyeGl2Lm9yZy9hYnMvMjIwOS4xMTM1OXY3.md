# CUTS: A Deep Learning and Topological Framework for Multigranular Unsupervised Medical Image Segmentation

Chen Liu, Matthew Amodio, Liangbo L. Shen, Feng Gao, Arman Avesta, Sanjay Aneja, Jay C. Wang, Lucian V. Del Priore, Smita Krishnaswamy (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 환자 진단과 정량적 연구에 필수적이지만, 이를 수행하기 위한 고품질의 라벨링 데이터(Labeled Data)를 확보하는 데에는 막대한 비용과 시간이 소요된다. 전문가의 주석(Annotation)은 노동 집약적일 뿐만 아니라 주석자 간의 일관성이 떨어지는 문제도 존재한다.

또한, 기존의 지도 학습(Supervised Learning) 기반 네트워크는 다음과 같은 한계를 가진다. 첫째, 데이터의 변동성을 충분히 커버하기 위해 대량의 라벨이 필요하다. 둘째, 특정 데이터셋으로 학습된 모델은 환자군이나 기기 환경이 약간만 달라져도 일반화 성능이 급격히 저하되는 Cross-domain generalization 문제가 발생한다. 셋째, 동일한 영상이라도 사용 목적에 따라 필요한 분할 수준(Granularity)이 다를 수 있는데(예: 뇌 종양 국소화 vs 뇌 전체 부피 측정), 지도 학습 방식은 라벨을 새로 업데이트하지 않고는 이러한 다양한 수준의 분할을 제공하기 어렵다.

본 논문의 목표는 전문가의 주석 없이도 다양한 수준의 세밀함(Multigranular)을 가진 분할 맵을 생성할 수 있는 완전히 비지도(Unsupervised) 방식의 딥러닝 프레임워크인 CUTS를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 전체의 전역적 특성보다는 **픽셀 중심 패치(Pixel-centered patch)**의 지역적 문맥(Local context)에 집중하여 표현 학습을 수행하고, 이를 **위상학적 데이터 조립(Topological data coarse graining)** 기법과 결합하는 것이다.

주요 기여 사항은 다음과 같다.
- **2단계 비지도 분할 프레임워크(CUTS)**: 첫 단계에서는 합성곱 인코더를 통해 픽셀 중심 패치를 고차원 임베딩 공간으로 매핑하고, 두 번째 단계에서는 Diffusion Condensation을 사용하여 이 임베딩들을 다양한 수준의 입도(Granularity)로 클러스터링하여 다중 스케일 분할을 수행한다.
- **새로운 최적화 목적 함수**: 이미지 내부의 대조 학습(Intra-image Contrastive Learning)과 지역 패치 재구성(Local Patch Reconstruction)을 결합하여 인코더가 더욱 표현력 있는 임베딩 공간을 학습하도록 설계하였다.
- **위상 기반 다중 스케일 클러스터링**: Diffusion Condensation을 도입하여 임상적으로 유의미한 다양한 스케일의 영역을 하이러키 구조로 추출함으로써, 임상의가 필요에 따라 적절한 세밀도의 분할 결과를 선택할 수 있게 하였다.

## 📎 Related Works

기존의 의료 영상 분할 방식은 크게 세 가지 흐름으로 나뉜다.
1. **전통적 방법**: 수작업으로 설계된 특징(Hand-crafted features)이나 미리 정의된 아틀라스(Atlas)를 사용한다. 하지만 복잡한 색상과 질감을 처리하는 데 한계가 있으며, 아틀라스 구축에 많은 시간이 소요된다.
2. **지도 학습(Supervised Learning)**: U-Net 및 그 변형 모델들이 뛰어난 성능을 보이지만, 대량의 전문가 주석이 필수적이며 도메인 일반화 능력이 부족하다.
3. **비지도 학습(Unsupervised Learning)**: 최근 DFC나 STEGO 같은 대조 학습 기반의 비지도 분할 방법들이 제안되었다. 특히 DFC는 본 연구와 유사하지만, 픽셀 단위로 대조를 수행한다는 점과 단일 분할 맵만을 생성한다는 점에서 차이가 있다. CUTS는 픽셀보다 풍부한 의미 정보를 가진 패치 단위를 사용하며, 위상학적 방법론을 통해 다중 스케일 분할을 제공한다.

또한, SAM(Segment Anything Model) 및 MedSAM과 같은 파운데이션 모델들이 등장하였으나, 이들은 거대한 라벨링 데이터셋을 통한 사전 학습에 의존하며, 여전히 도메인 특화된 미세 조정(Fine-tuning)이 필요하다는 한계가 있다.

## 🛠️ Methodology

CUTS 프레임워크는 크게 두 단계로 구성된다.

### 1. 픽셀 중심 패치 임베딩 학습 (Stage 1)
이미지의 각 픽셀을 중심으로 한 고정 크기의 패치(본 연구에서는 $5 \times 5$)를 추출하여 고차원 임베딩 공간으로 매핑하는 합성곱 인코더 $f$를 학습시킨다. 공간 해상도를 유지하기 위해 풀링(Pooling) 층은 사용하지 않는다. 학습을 위해 두 가지 손실 함수를 결합하여 사용한다.

**가. 이미지 내부 대조 손실 (Intra-image Contrastive Loss)**
이미지 내에서 앵커 패치 $P_{ij}$와 유사한 양성 패치(Positive patches, $\Omega^+$) 및 음성 패치(Negative patches, $\Omega^-$)를 샘플링한다. 양성 패치는 앵커 패치와 인접하고 구조적 유사도(SSIM)가 0.5보다 큰 패치들로 정의된다. 손실 함수 $l_{contrast}$는 다음과 같다.

$$l_{contrast} = -\log \frac{\sum_{z_{ij}^+ \in \Omega^+} e^{\text{sim}(z_{ij}, z_{ij}^+)/\tau}}{\sum_{z_{ij}^- \in \Omega^-} e^{\text{sim}(z_{ij}, z_{ij}^-)/\tau}}$$

여기서 $\text{sim}(\cdot)$은 코사인 유사도이며, $\tau$는 온도 파라미터이다.

**나. 지역 패치 재구성 손실 (Local Patch Reconstruction Loss)**
임베딩 벡터 $z_{ij}$가 주변 패치의 정보를 유지하도록 하기 위해, 임베딩으로부터 원래의 패치를 복원하는 재구성 모듈 $f_{recon}$을 학습시킨다.

$$l_{recon} = ||P_{ij} - f_{recon}(z_{ij})||_2^2$$

**다. 최종 목적 함수**
두 손실 함수를 가중치 $\lambda$를 이용해 결합한다.

$$\text{loss} = \lambda \cdot l_{contrast} + (1-\lambda) \cdot l_{recon}$$

### 2. 다중 스케일 분할을 위한 Coarse-graining (Stage 2)
학습된 임베딩 벡터들을 **Diffusion Condensation** 기법을 통해 클러스터링한다. 이 과정은 데이터 포인트들을 인접한 이웃 방향으로 반복적으로 응축시켜 자연스러운 그룹화를 찾아내는 동적 프로세스이다.

먼저 가우시안 커널을 이용해 데이터 포인트 간의 지역 친화도(Local Affinity) 행렬 $K$를 생성한다.
$$K(x_m, x_n) = e^{-\frac{||x_m - x_n||^2}{\epsilon}}$$

이후 확산 연산자(Diffusion Operator) $P = D^{-1}K$ (여기서 $D$는 차수 행렬)를 정의하고, 다음과 같이 반복적으로 데이터 행렬 $X$를 업데이트한다.
$$X_{t} = P_{t-1} X_{t-1}$$

이 프로세스를 통해 데이터는 점차 거친 입도(Coarse granularity)로 응축되며, 각 반복 단계(Iteration)마다 서로 다른 수준의 분할 맵을 얻을 수 있다. 특정 단계에서 클러스터가 유지되는 정도(Persistence)를 측정하여 위상학적으로 안정적인 구조를 식별하고, 이를 이미지 공간에 매핑하여 최종 분할 결과물을 생성한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 망막 저저색소침착(GA) 안저 이미지 56장, 뇌실(Ventricle) MRI 100장, 뇌종양(Tumor) MRI 200장.
- **비교 대상**: 
    - 전통적 비지도 방식: Watershed, Felzenszwalb, SLIC.
    - 딥러닝 비지도 방식: DFC, STEGO.
    - 파운데이션 모델: SAM, MedSAM, SAM-Med2D.
    - 지도 학습(상한선): UNet, nn-UNet.
- **평가 지표**: Dice Coefficient (DSC $\uparrow$), Hausdorff Distance (HD $\downarrow$).

### 주요 결과
1. **정량적 성능**: CUTS는 모든 데이터셋에서 기존 비지도 학습 방법들보다 최소 10% 이상의 성능 향상을 보였다. 특히 뇌종양 분할과 같이 대조가 미묘한 작업에서 다른 비지도 방법들이 실패하는 반면, CUTS는 유의미한 분할 성능을 유지하였다.
2. **SAM 계열과의 비교**: 단일 포인트 프롬프트를 사용한 조건에서, CUTS는 대규모 라벨 데이터로 사전 학습된 SAM 변형 모델들보다 3개 데이터셋 중 최소 2개에서 더 우수한 성능을 기록하였다.
3. **다중 스케일 특성**: 정성적 분석 결과, Diffusion Condensation의 반복 횟수에 따라 '미세 혈관 $\rightarrow$ 시신경 유두 $\rightarrow$ 전경/배경' 순으로 분할 수준이 변화하는 것을 확인하였다. 뇌 MRI의 경우 '종양 $\rightarrow$ 회백질/백질 $\rightarrow$ 뇌 추출(Brain Extraction)' 순으로 스케일이 확장되었다.
4. **Ablation Study**: 원본 픽셀 값만 사용하여 클러스터링을 수행한 경우보다 CUTS의 잠재 임베딩 공간을 사용했을 때 성능이 월등히 높음을 확인하여, 1단계 표현 학습의 중요성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상의 전역적 균질성(Global homogeneity)과 지역적 이질성(Local heterogeneity)이라는 특성에 주목하였다. 이미지 전체는 비슷해 보일 수 있지만, 조직의 경계나 병변 부위의 지역적 문맥은 뚜렷한 차이를 보인다는 점을 이용해 패치 기반의 대조 학습을 설계한 것이 주효하였다.

**강점 및 의의**:
- **라벨 독립성**: 전문가의 주석 없이도 작동하므로 희귀 질환과 같이 데이터 확보가 어려운 분야에 매우 유용하다.
- **유연한 분석**: 단일 마스크가 아닌 다중 스케일의 분할 맵을 제공함으로써, 임상의가 진단 목적에 따라 세밀도를 조절하며 분석할 수 있는 도구를 제공한다.
- **효율성**: 거대 모델의 사전 학습 없이도 특정 도메인의 데이터에서 효과적인 특징을 추출할 수 있는 경량화된 대안을 제시하였다.

**한계 및 논의**:
- 비지도 학습의 특성상, 최종 이진 마스크를 생성하기 위해 Ground Truth를 이용해 어떤 클러스터가 전경(Foreground)인지 선택하는 과정이 포함되어 있다. 실제 현장에서는 이 과정을 어떻게 자동화하거나 사용자 인터페이스로 구현할 것인지에 대한 추가 연구가 필요하다.
- Diffusion Condensation의 하이퍼파라미터($\epsilon$ 등)가 결과에 영향을 줄 수 있으며, 이에 대한 최적화 방법론이 더 구체적으로 다뤄질 필요가 있다.

## 📌 TL;DR

CUTS는 **'지역 패치 기반의 대조 학습'**과 **'위상학적 Diffusion Condensation'**을 결합한 비지도 의료 영상 분할 프레임워크이다. 이 연구는 라벨링 데이터 없이도 정밀한 분할이 가능함을 보였으며, 특히 다양한 세밀도(Multigranular)의 분할 맵을 생성함으로써 임상적 유연성을 확보하였다. 이는 대규모 데이터셋에 의존하는 현재의 파운데이션 모델 트렌드 속에서, 도메인 특화된 귀납적 편향(Inductive bias)을 주입하는 비지도 학습의 중요성을 재확인시켜 준 연구이다.