# CMamba: Learned Image Compression with State Space Models

Zhuojie Wu, Heming Du, Shuyun Wang, Ming Lu, Haiyang Sun, Yandong Guo, Xin Yu (2021/2025)

## 🧩 Problem to Solve

본 논문은 Learned Image Compression (LIC) 분야에서 고성능의 Rate-Distortion (RD) 성능을 달성하면서 동시에 계산 복잡도(파라미터 수, FLOPs, 지연 시간)를 낮게 유지하는 문제를 해결하고자 한다.

기존의 LIC 모델들은 이미지 콘텐츠 분포를 모델링하기 위해 Convolutional Neural Networks (CNNs)와 Transformers를 주로 사용해 왔다. 그러나 CNNs는 전역적 문맥(Global Context)을 캡처하는 능력이 부족하여 latent representation에 중복성이 발생하며, Transformers는 강력한 long-range modeling 능력을 갖추었으나 self-attention의 이차 복잡도(Quadratic Complexity)로 인해 계산 비용이 매우 높다는 한계가 있다.

따라서 본 논문의 목표는 State Space Models (SSMs)의 선형 복잡도와 전역 수용 영역(Global Receptive Field)의 장점을 활용하여, 효율적이면서도 강력한 성능을 내는 하이브리드 이미지 압축 프레임워크인 CMamba를 제안하는 것이다.

## ✨ Key Contributions

CMamba의 핵심 아이디어는 SSM의 전역 모델링 능력과 CNN의 국소 세부 정보 캡처 능력을 결합하는 하이브리드 설계에 있다.

1. **Content-Adaptive SSM (CA-SSM) 모듈**: SSM은 전역적 내용은 잘 파악하지만 고주파 세부 정보(High-frequency details)를 손실하는 경향이 있다. 이를 보완하기 위해 SSM과 CNN을 병렬로 배치하고, 두 특징을 동적으로 융합하여 이미지의 전역적 문맥과 국소적 디테일을 모두 보존한다.
2. **Context-Aware Entropy (CAE) 모듈**: latent representation의 공간적 및 채널 간 중복성을 동시에 제거한다. 공간적 차원에서는 SSM을 사용하여 효율적으로 분포를 파라미터화하고, 채널 차원에서는 자기회귀(Autoregressive) 방식을 통해 이전 채널의 정보를 활용함으로써 압축 효율을 극대화한다.

## 📎 Related Works

**1. 이미지 압축 (Image Compression)**
JPEG, BPG, VVC와 같은 전통적인 손실 압축 표준은 수작업으로 설계된 규칙에 의존한다. 최근의 LIC 방법들은 end-to-end 최적화를 통해 더 나은 RD 성능을 보이고 있으며, 특히 Transformer 기반 모델들이 뛰어난 성능을 보였으나 높은 계산 비용이 문제로 지적되었다.

**2. 상태 공간 모델 (State Space Models, SSMs)**
SSMs는 선형 시간 복잡도로 시퀀스 데이터를 모델링할 수 있는 모델이다. Mamba는 selective scanning과 하드웨어 가속을 통해 효율성을 높였으며, Vim과 VMamba는 이를 시각 데이터에 적용하여 비전 백본으로서의 가능성을 보여주었다. 그러나 이미지 압축 분야에서 SSM을 어떻게 효과적으로 적용할지에 대해서는 기존 연구가 부족했다.

## 🛠️ Methodology

### 1. 전체 파이프라인

CMamba는 일반적인 LIC의 3단계 패러다임인 비선형 변환, 양자화, 엔트로피 코딩을 따른다.

- **분석 변환 ($g_a$)**: 입력 이미지 $x$를 compact한 latent representation $y$로 매핑한다.
- **양자화 ($Q$)**: $y$를 이산 값으로 변환한다. 이때 발생하는 양자화 오차 $r = y - Q(y)$는 LRP(Latent Residual Prediction) 네트워크를 통해 추정된다.
- **합성 변환 ($g_s$)**: 보정된 latent $\bar{y} = \hat{y} + r$을 다시 이미지 $\hat{x}$로 복원한다.

전체 학습 목표는 다음과 같은 Rate-Distortion 손실 함수를 최소화하는 것이다.
$$L = R(\hat{y}) + R(\hat{z}) + \lambda \cdot D(x, \hat{x})$$
여기서 $R$은 비트레이트, $D$는 왜곡도(MSE 또는 MS-SSIM), $\lambda$는 두 값 사이의 트레이드-오프를 조절하는 하이퍼파라미터이다.

### 2. Content-Adaptive SSM (CA-SSM) 모듈

CA-SSM은 전역 문맥을 잡는 VSS(Visual State Space) 블록과 국소 디테일을 잡는 ResBlock을 병렬로 구성한다.

- **VSS 블록**: SS2D(2D-Selective-Scan) 레이어를 사용하여 이미지를 4가지 방향의 시퀀스로 변환하고 SSM으로 처리한 뒤 다시 병합하여 전역 특징 $F^{SSM}$을 추출한다.
- **ResBlock**: CNN 기반의 잔차 블록을 통해 고주파 세부 정보인 국소 특징 $F^{CNN}$을 추출한다.
- **Dynamic Fusion**: 두 특징을 단순히 더하는 것이 아니라, Global Max Pooling과 MLP를 거쳐 계산된 어텐션 가중치 $\alpha, \beta$를 이용해 동적으로 융합한다.
$$y = w(\alpha \cdot F^{SSM} + \beta \cdot F^{CNN})$$

### 3. Context-Aware Entropy (CAE) 모듈

CAE는 latent representation $y$의 확률 분포를 정밀하게 모델링하여 비트수를 줄인다.

- **공간적 의존성**: SSM을 활용하여 가우시안 모델의 파라미터를 추정한다. SSM의 선형 복잡도 덕분에 전역적 의존성을 효율적으로 모델링할 수 있다.
- **채널 간 의존성**: 채널을 순차적으로 처리하는 자기회귀(Autoregressive) 방식을 채택한다. 현재 채널 $y_i$의 분포 파라미터 $\Phi_i$를 추정할 때, 하이퍼프라이어 $\Phi'$와 이전에 디코딩된 그룹 $\bar{y}_{<i}$를 조건으로 사용한다.
$$\Phi_i = w_{ffn}(LN(F^{SSM})) + F^{SSM}, \quad \text{where } F^{SSM} = f_{ssm}(w_{sq}([\Phi', \bar{y}_{<i}])) + w_{sq}([\Phi', \bar{y}_{<i}])$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Kodak, Tecnick, CLIC 데이터셋을 사용하였다.
- **평가 지표**: PSNR, MS-SSIM, BPP(bits per pixel) 및 BD-Rate를 사용하여 성능을 측정하였다. BD-Rate는 VVC를 기준으로 설정하여 상대적인 비트레이트 절감량을 나타낸다.

### 2. 정량적 결과

CMamba는 모든 벤치마크 데이터셋에서 기존의 SOTA 방법 및 전통적 코덱인 VVC보다 우수한 RD 성능을 보였다.

- **BD-Rate 개선 (vs VVC)**: Kodak(-14.95%), Tecnick(-18.83%), CLIC(-13.89%)로 비트레이트를 크게 절감하였다.
- **계산 효율성 (vs MLIC++)**: Kodak 데이터셋 기준, 파라미터 수는 51.8% 감소, FLOPs는 28.1% 감소, 디코딩 시간은 71.4% 감소하는 획기적인 효율 향상을 달성하였다.

### 3. 정성적 결과 및 분석

시각적 비교 결과, CMamba는 TCM 등의 모델보다 더 낮은 비트레이트에서도 발코니 난간의 질감이나 벽화의 세부 묘사와 같은 고주파 디테일을 더 날카롭게 보존하는 것으로 나타났다. 또한, 공간 상관관계 맵(Spatial Correlation Map) 분석을 통해 CMamba가 latent representation의 중복성을 가장 효과적으로 제거하여 decorrelation을 달성했음을 입증하였다.

## 🧠 Insights & Discussion

**강점 및 기여**
본 논문은 SSM이 가진 전역 모델링 능력과 선형 복잡도라는 강력한 장점을 이미지 압축에 성공적으로 도입하였다. 특히, SSM의 고질적인 문제인 고주파 정보 손실을 CNN과의 동적 융합(CA-SSM)으로 해결한 점이 매우 영리한 접근이다. 또한, 엔트로피 모델(CAE)에서 공간-채널 의존성을 동시에 고려함으로써 압축 효율을 극대화하면서도 추론 속도를 유지하였다.

**한계 및 논의**
본 연구는 latent representation이 가우시안 분포를 따른다고 가정하고 모델링하였다. 실제 데이터의 분포가 더 복잡할 경우, 단순 가우시안 모델 이상의 정교한 확률 모델(예: Mixture of Gaussians)을 SSM과 결합한다면 추가적인 성능 향상이 있을 수 있을 것이다. 또한, 제시된 결과는 매우 긍정적이나, 다양한 해상도의 이미지나 극단적인 저비트레이트 환경에서의 강건성에 대한 추가 분석이 필요해 보인다.

## 📌 TL;DR

CMamba는 **SSM의 선형 복잡도/전역 수용 영역**과 **CNN의 국소 디테일 캡처 능력**을 결합한 하이브리드 이미지 압축 프레임워크이다. **CA-SSM**을 통해 전역-국소 특징을 동적으로 융합하고, **CAE** 모듈로 공간 및 채널 중복성을 효율적으로 제거함으로써, VVC 및 기존 SOTA LIC 모델 대비 **더 뛰어난 RD 성능**을 보이면서도 **파라미터 수와 디코딩 시간을 획기적으로 줄였다**. 이 연구는 향후 효율적인 딥러닝 기반 이미지/비디오 압축 모델 설계에 있어 SSM이 매우 유망한 대안이 될 수 있음을 시사한다.
