# Privacy-Preserving Collaborative Medical Image Segmentation Using Latent Transform Networks

Saheed Ademola Bello, Muhammad Shahid Jabbar, Muhammad Sohail Ibrahim, Shujaat Khan (2026)

## 🧩 Problem to Solve

현대 의료 영상 분석에서 딥러닝 기반의 세그멘테이션(Segmentation) 모델을 구축하기 위해서는 대규모의 다양하고 정교하게 주석 처리된 데이터셋이 필수적이다. 그러나 실제 의료 환경에서는 엄격한 개인정보 보호 규정, 병원 간의 데이터 사일로(Data Silos) 현상, 그리고 데이터 가용성의 불균형으로 인해 원본 의료 영상이나 마스크(Annotation) 데이터를 외부로 공유하는 것이 불가능하다. 이는 모델의 일반화 성능을 저하시키고 특정 인구 집단이나 장비에 편향된 모델을 생성하는 결과로 이어진다.

기존의 해결책인 연합 학습(Federated Learning, FL)은 원본 데이터를 공유하지 않지만, 그래디언트 역전(Gradient Inversion) 및 멤버십 추론 공격(Membership Inference Attacks)에 취약하며 빈번한 통신 오버헤드가 발생한다. 동형 암호(Homomorphic Encryption)와 같은 암호화 방식은 이론적으로는 강력하나 계산 비용이 너무 커 실시간 적용이 어렵다. 최근 제안된 latent-space 기반의 Privacy-SF 프레임워크는 통신 비용을 줄였으나, 저해상도 병목(Bottleneck) 구조로 인해 세밀한 공간 정보와 경계선 디테일이 손실되는 성능 저하 문제와 latent inversion 공격에 취약하다는 한계가 있다. 따라서 본 논문은 **데이터 프라이버시를 강력하게 보호하면서도 세그멘테이션의 정확도(특히 경계선 정밀도)를 유지하는 협력적 의료 영상 세그멘테이션 프레임워크**를 개발하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Skip-connected Autoencoders**를 통해 공간적 디테일을 보존하고, **Keyed Latent Transform (KLT)**를 통해 전송되는 잠재 특징(Latent features)을 암호화하여 프라이버시를 보호하는 것이다.

1. **PPCMI-SF 프레임워크 제안**: Skip-connection이 적용된 오토인코더와 KLT를 결합하여, 보안성이 강화된 latent-space 협력 학습 구조를 설계하였다.
2. **Unified Mapping Network (UMN) 설계**: Pyramid Pooling Module(PPM)과 역전된 인코더-디코더 계층 구조를 도입하여, 프라이버시 제약 조건 하에서도 latent-to-latent 변환의 정확도를 극대화하였다.
3. **프라이버시 강건성 검증**: Cross-decoder inversion 및 멤버십 추론 공격 실험을 통해 제안 방법론이 외부 공격에 대해 강력한 저항력을 가짐을 입증하였다.
4. **다양한 모달리티에 대한 일반화 성능 입증**: 초음파(Ultrasound), CT, MRI 등 서로 다른 의료 영상 데이터셋에서 프라이버시 보호 모델임에도 불구하고 비보호(Privacy-agnostic) 베이스라인에 근접하는 성능을 보였다.

## 📎 Related Works

논문에서는 프라이버시 보호 세그멘테이션을 위한 기존 접근 방식을 다음과 같이 분류하고 한계를 지적한다.

- **연합 학습 및 블록체인 기반**: 원본 데이터를 로컬에 유지하지만, 그래디언트 유출 및 멤버십 추론 공격에 취약하며 통신 오버헤드가 크다.
- **암호화 및 차분 프라이버시 (Differential Privacy)**: 강력한 이론적 보장을 제공하지만, 계산 비용이 매우 높거나(HE/SMPC), 노이즈 주입으로 인해 세그멘테이션 정확도가 하락하는 트레이드-오프가 발생한다.
- **합성 데이터 및 지속 학습**: 직접적인 데이터 공유를 피할 수 있으나, 합성 데이터가 실제 해부학적 구조를 왜곡할 위험이 있으며 복잡한 모달리티에서 정확도가 떨어진다.
- **인코딩 및 매핑 기반 (Encoding-and-Mapping)**: 데이터를 잠재 공간으로 압축하여 전송하므로 효율적이지만, 기존의 Privacy-SF 등은 저해상도 병목 현상으로 인해 공간적 세부 정보가 손실되고 latent-feature inversion 공격에 취약하다는 단점이 있다.

PPCMI-SF는 이러한 한계를 극복하기 위해 **Skip-connection을 통한 특징 재사용**과 **직교 행렬 기반의 KLT를 이용한 잠재 공간 난독화**를 통해 성능과 보안성을 동시에 잡고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

PPCMI-SF는 클라이언트-서버 구조로 동작하며, 원본 데이터는 절대 서버로 전송되지 않는다.

- **클라이언트 사이드**: 영상 오토인코더($E_x, D_x$)와 마스크 오토인코더($E_y, D_y$)를 독립적으로 학습시키고, 추출된 latent feature에 KLT를 적용하여 서버로 전송한다.
- **서버 사이드**: 수신된 보호된 latent feature를 역변환하여 공유 도메인으로 복원한 뒤, UMN을 통해 영상 latent를 마스크 latent로 매핑하고, 다시 KLT를 적용하여 클라이언트로 반환한다.

### 2. Keyed Latent Transform (KLT)

KLT는 잠재 텐서 $z \in \mathbb{R}^{B \times C \times H \times W}$를 다음과 같이 변환한다.
$$z' = T(z) = Q^\top z + b$$
여기서 $Q \in \mathbb{R}^{C \times C}$는 QR 분해를 통해 생성된 클라이언트별 고유 직교 행렬(Orthogonal Matrix)이며, $b \in \mathbb{R}^C$는 바이어스 벡터이다. 역변환은 다음과 같다.
$$z = T^{-1}(z') = Q(z' - b)$$
이 변환은 선형성과 미분 가능성을 유지하므로 서버에서의 학습을 방해하지 않으면서도, 외부 공격자가 $Q$와 $b$를 모를 경우 원래의 latent feature를 복원하는 것을 어렵게 만든다.

### 3. Unified Mapping Network (UMN)

UMN은 서버에서 $\hat{T}(z_y) = T \circ M_{\text{shared}} \circ T^{-1} \circ T(z_x)$ 연산을 수행한다.

- **구조적 특징**: 일반적인 인코더-디코더와 달리 **역전된 계층 구조**를 가진다. 즉, 먼저 bilinear interpolation로 업샘플링하여 공간 해상도를 확장한 뒤, 나중에 max-pooling으로 축소함으로써 정보 손실을 최소화한다.
- **Pyramid Pooling Module (PPM)**: $1\times1, 3\times3, 5\times5, 7\times7$의 네 가지 스케일로 풀링을 수행하여 전역 문맥 정보(Global Context)를 캡처하고 이를 기본 특징 맵과 결합한다.

### 4. 학습 절차 및 손실 함수

학습은 2단계로 진행된다.

- **Step 1 (클라이언트)**: 오토인코더를 로컬 데이터로 학습시킨다. 픽셀 단위 재구성 정확도와 영역 구조 일관성을 위해 BCE와 Dice Loss를 결합하여 사용한다.
    $$\mathcal{L}_{AE} = \alpha \cdot \text{BCE}(u, \hat{u}) + (1-\alpha) \cdot \text{Dice}(u, \hat{u})$$
- **Step 2 (서버)**: 보호된 latent pair $[T(z_x), T(z_y)]$를 받아 UMN을 학습시킨다. 손실 함수로는 다중 스케일 평균 제곱 오차(MSE)를 사용한다.
    $$\mathcal{L}_{map} = \sum_{\ell \in \{1,2,3\}} \| z_{y\ell} - \hat{z}_{y\ell} \|^2$$

## 📊 Results

### 1. 데이터셋 및 지표

- **데이터셋**: PSFH(초음파), US Nerve(초음파), Cardiac MRI, FUMPE(CTA)의 4개 데이터셋을 사용하였다.
- **측정 지표**: 영역 중첩도를 측정하는 Dice Similarity Coefficient (DSC)와 경계선 정확도를 측정하는 95th percentile Hausdorff Distance (HD95), Average Symmetric Surface Distance (ASD)를 사용하였다.

### 2. 정량적 결과 (PSFH 데이터셋)

PPCMI-SF는 기존 Privacy-SF 대비 괄목할 만한 성능 향상을 보였다.

- **DSC**: $87.60 \pm 0.07 \to 90.49 \pm 0.05$로 상승하였다.
- **경계선 정밀도**: HD95와 ASD 값이 크게 낮아져, 경계선 묘사 능력이 향상되었음을 확인하였다.
- **비보호 모델 대비**: UNet, nnUNetv2 등 원본 데이터를 직접 사용하는 모델들과 비교했을 때, 매우 근소한 차이의 성능을 유지하면서도 프라이버시를 보장하였다.

### 3. 일반화 및 강건성 테스트

- **Cross-modality**: FUMPE CTA 데이터셋에서는 오히려 기존 CNN 베이스라인보다 높은 DSC(74.54%)를 기록하며 뛰어난 일반화 능력을 보였다.
- **프라이버시 공격**:
  - **Cross-decoder Inversion**: 타 클라이언트의 디코더로 복원 시도 시, Privacy-SF는 일부 구조가 복원(SSIM 0.69)되었으나, PPCMI-SF는 완전히 왜곡된 이미지(SSIM 0.34)를 생성하여 강력한 저항력을 보였다.
  - **Membership Inference (MIA)**: AUC 값이 0.473으로 무작위 추측(0.5)에 매우 가깝게 나타나, 학습 데이터 포함 여부를 판별할 수 없음을 입증하였다.

### 4. 효율성 분석

- **추론 시간**: 전체 엔드-투-엔드 지연 시간은 약 $19.07\text{ms}$로 실시간 적용이 가능하다.
- **통신 비용**: 쿼리당 전송 데이터 양은 약 $0.88\text{MB}$로 매우 낮다.

## 🧠 Insights & Discussion

본 연구는 의료 영상의 프라이버시 보호와 세그멘테이션 성능이라는 두 마리 토끼를 잡기 위해 **'잠재 공간의 난독화'**와 **'공간 정보의 보존'**이라는 전략을 취하였다.

**강점 및 통찰**:

- **KLT의 역할**: Ablation study 결과, KLT는 세그멘테이션 정확도에는 영향을 주지 않지만(이미 서버에서 역변환 후 학습하므로), 전송 단계에서의 프라이버시 보호와 MIA 공격 방어에는 결정적인 역할을 한다는 점이 밝혀졌다.
- **Skip-connection의 중요성**: 기존 Privacy-SF의 성능 저하 원인이 단순한 병목 구조에 있었음을 확인하였으며, Skip-connection 도입만으로도 잠재 공간 내에서 충분한 공간 정보를 유지할 수 있음을 보여주었다.
- **UMN의 설계**: 일반적인 다운샘플링 구조가 아닌 업샘플링 우선 구조를 통해 latent-to-latent 매핑 시 발생하는 정보 손실을 억제한 점이 유효했다.

**한계 및 논의**:

- 본 모델은 서버가 '정직하지만 호기심 많은(Honest-but-curious)' 상태라고 가정하며, KLT 키($Q, b$)가 유출되지 않는다는 전제하에 작동한다. 만약 키가 탈취되거나 서버가 완전히 악의적일 경우의 보안 대책은 아직 미비하다.
- 따라서 향후 연구에서는 TEE(Trusted Execution Environment)와의 통합이나 적응형 키 로테이션(Adaptive Key Rotation) 기법을 도입하여 보안성을 더욱 강화할 필요가 있다.

## 📌 TL;DR

본 논문은 의료 영상의 프라이버시를 보호하면서 여러 기관이 협력하여 세그멘테이션 모델을 학습할 수 있는 **PPCMI-SF** 프레임워크를 제안한다. **Skip-connected Autoencoders**로 세부 공간 정보를 보존하고, **직교 행렬 기반의 Keyed Latent Transform (KLT)**로 데이터를 암호화하여 전송함으로써, 데이터 유출 위험을 최소화하면서도 비보호 모델에 근접하는 높은 세그멘테이션 정확도를 달성하였다. 특히 latent inversion 및 멤버십 추론 공격에 대해 강력한 저항성을 보였으며, 실시간 추론이 가능한 효율성을 갖추어 실제 의료 현장의 다기관 협력 학습에 적용될 가능성이 매우 높다.
