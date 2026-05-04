# SALT: Parameter-Efficient Fine-Tuning via Singular Value Adaptation with Low-Rank Transformation

Abdelrahman Elsayed et al. (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 해부학적 구조나 병리적 영역을 정밀하게 구분해야 하므로, 도메인 특화된 세부 특징을 포착할 수 있는 모델 설계가 필수적이다. 최근 Segment Anything Model(SAM)과 같은 거대 기초 모델(Foundation Model)이 등장하며 유연한 적용 가능성을 보여주었으나, 이를 의료 데이터에 맞게 미세 조정(Fine-tuning)하는 비용은 매우 높은 진입 장벽으로 작용한다.

기존의 매개변수 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 방법론들은 다음과 같은 한계를 가진다. 첫째, Low-Rank Adaptation(LoRA)은 저차원 행렬을 통해 가중치를 업데이트하지만, 선택된 랭크(rank)가 너무 낮을 경우 도메인 특유의 세밀한 특징을 포착하지 못해 과소적합(Underfitting) 문제가 발생할 수 있다. 둘째, 특이값 분해(Singular Value Decomposition, SVD) 기반 방법론들은 모든 특이값을 수정하여 포괄적인 업데이트를 제공하지만, 유연성이 부족하고 데이터셋에 따라 성능 편차가 크게 나타난다.

본 논문의 목표는 이러한 LoRA의 표현력 부족 문제와 SVD 기반 방법의 유연성 부족 문제를 동시에 해결하여, 최소한의 매개변수 업데이트만으로도 의료 영상 도메인에 효과적으로 적응할 수 있는 PEFT 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **SVD 기반의 지배적 특이값 적응(Dominant Singular Value Adaptation)과 저차원 잔차 업데이트(Low-Rank Residual Updates)를 결합**하는 하이브리드 접근 방식인 SALT를 제안하는 것이다.

SALT는 가중치 행렬을 SVD로 분해한 후, 영향력이 큰 상위 특이값들은 학습 가능한 스케일(scale) 및 시프트(shift) 파라미터를 통해 효율적으로 조정하고, 나머지 하위 공간(subspace)에 대해서는 LoRA 방식의 저차원 변환을 적용한다. 이를 통해 기초 모델이 가진 일반적인 지식을 보존하면서도, 의료 영상의 특수한 세부 사항을 정밀하게 학습할 수 있는 균형 잡힌 적응 능력을 확보한다.

## 📎 Related Works

논문에서는 SAM을 의료 영상에 적용하려는 MedSAM, SAM Adapter, S-SAM 등의 연구를 언급한다. 특히 PEFT 관점에서 LoRA와 AdaLoRA는 가중치 업데이트를 저차원 행렬로 근사하여 효율성을 높였으나, 지배적인 특징과 미세한 특징을 동시에 포착하는 데 어려움이 있음을 지적한다.

SVD 기반의 PiSSA나 S-SAM 같은 방식은 주요 특이 벡터에 집중하여 데이터 패턴을 포착하려 하지만, 이러한 경직된 우선순위 지정 방식은 동적이거나 매우 세밀한 데이터 특성에 대응하는 적응력을 제한한다는 한계가 있다. SALT는 이러한 기존 방식들의 절충안으로서, 중요한 정보는 SVD 스케일링으로, 세부 정보는 LoRA 업데이트로 처리함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
SALT는 SAM의 이미지 인코더(Image Encoder) 내에 위치한 Multi-Head Attention(MHA) 층과 MLP 층의 가중치 행렬 $W^{(l)}$을 대상으로 적용된다. 전체 모델의 가중치는 동결(frozen)시킨 상태에서 SALT 모듈의 파라미터와 Normalization Layer, Text Affine Layer만을 학습시킨다.

### 상세 방법론 및 방정식
1. **SVD 분해 및 분할**: 우선 가중치 행렬 $W^{(l)}$을 다음과 같이 SVD 분해한다.
   $$W^{(l)} = U^{(l)}\Sigma^{(l)}(V^{(l)})^\top$$
   여기서 $\Sigma^{(l)}$은 특이값들로 구성된 대각 행렬이다. SALT는 이를 상위 $r$개의 특이값을 가진 $\Sigma^{(l)}_r$과 나머지 $r'$개의 특이값을 가진 $\Sigma^{(l)}_{r'}$로 분리한다.

2. **상위 특이값 적응 (Scale & Shift)**: 상위 $r$개의 특이값에 대해 학습 가능한 대각 행렬 $\alpha^{(l)}$(스케일)과 $\beta^{(l)}$(시프트)를 적용하여 다음과 같이 변환한다.
   $$\Sigma'^{(l)}_r = \alpha^{(l)} \odot \Sigma^{(l)}_r + \beta^{(l)}$$
   (여기서 $\odot$은 Hadamard product를 의미한다.)

3. **하위 공간 적응 (Low-Rank Update)**: 나머지 $\Sigma^{(l)}_{r'}$ 부분에 대해서는 LoRA와 유사한 저차원 행렬 곱 $X^{(l)}Y^{(l)\top}$를 적용한다. 이때 $X^{(l)} \in \mathbb{R}^{r' \times d_{lora}}$이고 $Y^{(l)} \in \mathbb{R}^{d_{lora} \times r'}$이다.

4. **가중치 재구성**: 변환된 특이값 행렬 $\Sigma'^{(l)}$에 ReLU를 적용하여 반양정치성(semi-definiteness)을 유지한 후, 최종 가중치 $\tilde{W}^{(l)}$를 다음과 같이 계산한다.
   $$\tilde{W}^{(l)} = U^{(l)} \text{ReLU}(\Sigma'^{(l)}) (V^{(l)})^\top$$

### 학습 목표 및 손실 함수
SALT의 학습은 Focal Loss와 Dice Loss를 기본으로 하며, 추가적으로 다음과 같은 정규화 항(Regularization term)을 포함한 손실 함수 $L_{reg}$를 사용한다.
$$L_{reg} = \underbrace{\|\alpha \odot s + \beta - s\|_F}_{\text{Scale-Shift}} + \underbrace{\|XY\|_F}_{\text{LoRA}}$$
이 식에서 $\|\cdot\|_F$는 Frobenius norm을 의미하며, 이는 사전 학습된 특징을 최대한 유지하면서 최소한의 수정만을 가하도록 강제하는 역할을 한다.

## 📊 Results

### 실험 설정
- **데이터셋**: DIAS(뇌혈관), ROSE(망막 OCT-A), DRIVE(망막 RGB), ARCADE 및 XRay-Angio(관상동맥 X-ray) 등 총 5개의 의료 영상 데이터셋을 사용하였다. 표본 수는 20개에서 1000개까지 다양하게 구성되어 저리소스 환경에서의 성능을 함께 평가하였다.
- **지표**: Dice Similarity Coefficient (DSC)와 95th percentile Hausdorff Distance (HD95)를 사용하였다.
- **비교 대상**: 전통적인 DL 모델(U-Net, UNETR, DeepLabV3+ 등) 및 SAM 기반 PEFT 방법(LoRA, S-SAM)과 비교하였다.

### 정량적 결과
- **성능 향상**: SALT는 랭크 256 설정에서 평균 Dice 점수 0.74를 기록하며 LoRA와 S-SAM을 2%~5% 상회하는 성능을 보였다. 특히 DIAS(0.71), ROSE(0.67), XRay-Angio(0.77), Drive(0.75)에서 가장 높은 점수를 기록하였다.
- **효율성**: 전체 파라미터의 단 3.9%만을 학습시키고도 전파라미터 미세 조정 모델이나 다른 PEFT 방법보다 우수한 성능을 달성하였다.
- **경계 정밀도**: HD95 지표에서 SALT는 평균 23.87을 기록하여 LoRA(25.94) 및 S-SAM(30.12)보다 낮은 오차를 보였으며, 이는 특히 저대조도 망막 영상(DRIVE)에서 정밀한 혈관 묘사 능력이 뛰어남을 입증한다.

### 정성적 결과
정성적 분석 결과, SALT는 다른 방법론들에 비해 혈관의 복잡한 분지 구조(branching patterns)와 그물망 형태의 패턴을 더 정확하게 포착하였으며, 위양성(False Positive) 발생 빈도가 현저히 낮음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
SALT의 성공 요인은 SVD를 통해 가중치의 '중요도'를 먼저 정의하고, 중요한 성분은 직접적으로 스케일링하고 덜 중요한 성분은 저차원 근사로 처리한 전략에 있다. 이는 모델이 가진 기반 지식을 파괴하지 않으면서도 의료 도메인의 미세한 특성을 학습할 수 있게 한다. 특히 SAM2로 확장 적용했을 때, LoRA보다 훨씬 적은 파라미터(랭크 256 기준 2.4배 적음)로 경쟁력 있는 성능을 낸 점은 SALT의 효율성을 뒷받침한다.

### 한계 및 논의사항
- **랭크 선택의 민감도**: Ablation Study 결과, MHA 및 MLP 층의 랭크 설정(예: 700 $\rightarrow$ 200으로 감소)이 성능에 영향을 미친다는 점이 밝혀졌다. 이는 최적의 성능을 내기 위해 각 레이어별로 적절한 랭크를 수동으로 설정해야 하는 부담이 있음을 시사한다.
- **가정**: 본 연구는 SVD를 통해 상위 특이값이 항상 가장 중요한 정보를 담고 있다는 가정 하에 작동한다. 하지만 도메인 전이가 극심한 경우, 하위 특이값이 오히려 중요한 역할을 할 가능성에 대해서는 명시적으로 논의되지 않았다.

## 📌 TL;DR

본 논문은 SAM과 같은 거대 모델을 의료 영상 분할 태스크에 효율적으로 적응시키기 위해, **상위 특이값의 스케일/시프트 조정과 하위 공간의 저차원 업데이트(LoRA)를 결합한 SALT** 프레임워크를 제안한다. 실험 결과, SALT는 단 3.9%의 학습 가능한 파라미터만으로 기존 PEFT 방법론들보다 높은 Dice 점수와 낮은 HD95 오차를 기록하였다. 이는 기초 모델의 범용적 지식 보존과 도메인 특화 적응 사이의 균형을 효과적으로 맞춘 설계이며, 향후 저리소스 의료 영상 분석 및 타 기초 모델 적응 연구에 중요한 기여를 할 것으로 보인다.