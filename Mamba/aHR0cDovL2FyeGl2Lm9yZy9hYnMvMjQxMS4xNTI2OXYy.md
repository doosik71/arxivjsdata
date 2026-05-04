# MambaIRv2: Attentive State Space Restoration

Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, Yawei Li (2025)

## 🧩 Problem to Solve

본 논문은 최근 이미지 복원(Image Restoration) 분야에서 전역 수용 영역(Global Reception)과 계산 효율성의 균형을 맞추기 위해 도입된 Mamba 기반 백본의 한계점을 해결하고자 한다. Mamba의 핵심적인 문제는 **인과적 모델링(Causal Modeling)** 특성에 있다. Mamba는 스캔된 시퀀스 내에서 각 토큰이 오직 이전의 토큰들에만 의존하도록 설계되어 있어, 이미지 복원과 같은 비인과적(Non-causal) 작업에서 다음과 같은 세 가지 주요 문제점을 야기한다.

첫째, **제한된 지각 능력(Constrained Perception)**이다. 쿼리 픽셀이 스캔 순서상 이후에 오는 픽셀들의 정보를 활용할 수 없으므로, 이미지 전체에 걸쳐 존재하는 유용한 픽셀들을 충분히 활용하지 못한다. 둘째, 이를 극복하기 위해 기존 연구들은 **다방향 스캔(Multi-directional Scans)** 방식을 채택했으나, 이는 고해상도 입력 시 계산 복잡도를 크게 증가시키며 스캔 간의 상당한 정보 중복을 초래한다. 셋째, **장거리 감쇠(Long-range Decay)** 문제이다. Mamba의 제어 행렬 $A$의 값이 통계적으로 1보다 작기 때문에, 시퀀스 상에서 거리가 먼 픽셀 간의 상호작용은 급격히 약화되어 멀리 떨어져 있지만 의미적으로 유사한 픽셀들을 효과적으로 활용할 수 없다.

결과적으로 본 논문의 목표는 Mamba에 Vision Transformer(ViT)와 유사한 **비인과적 모델링 능력**을 부여하여, 효율성을 유지하면서도 전역적인 픽셀 상호작용을 극대화하는 'Attentive State Space Restoration' 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 상태 공간 방정식(State Space Equation)과 Attention 메커니즘 사이의 수학적 유사성을 이용해, Mamba가 스캔 시퀀스를 넘어 전역적으로 정보를 쿼리할 수 있게 만드는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Attentive State-space Equation (ASE) 제안**: 프롬프트 학습(Prompt Learning)을 Mamba의 출력 행렬 $C$에 결합하여, 스캔되지 않은 영역의 픽셀들까지 전역적으로 참조할 수 있는 비인과적 구조를 설계하였다. 이를 통해 다방향 스캔 없이 단 한 번의 스캔만으로도 전역 지각 능력을 확보하여 효율성을 높였다.
2.  **Semantic Guided Neighboring (SGN) 제안**: Mamba의 장거리 감쇠 문제를 해결하기 위해, 공간적 순서가 아닌 의미적 유사성에 따라 픽셀을 재배치하는 구조를 도입하였다. 의미적으로 유사한 픽셀들을 1D 시퀀스 상에서 가깝게 배치함으로써, 물리적으로 멀리 떨어진 픽셀 간의 상호작용을 강화하였다.
3.  **MambaIRv2 아키텍처 구현**: ASE와 SGN을 통합하고, 국소적 상호작용을 위한 Window Multi-Head Self-Attention(MHSA)을 결합하여, 전역-국소 계층 구조를 갖춘 고성능 이미지 복원 백본을 완성하였다.

## 📎 Related Works

이미지 복원 분야는 초기 CNN 기반 방법론(SRCNN, EDSR, RCAN 등)에서 시작하여, 전역 수용 영역을 확보하기 위해 ViT 기반 방법론(SwinIR, HAT 등)으로 발전하였다. 그러나 ViT의 Self-attention은 입력 크기에 따라 계산 복잡도가 제곱으로 증가하는 한계가 있다. 이를 해결하기 위해 최근 Mamba와 같은 선택적 상태 공간 모델(Selective State Space Model)이 대안으로 제시되었으며, MambaIR 등이 그 시작점이 되었다.

기존 Mamba 기반 방법론들은 2D 이미지를 1D 시퀀스로 펼치기 위해 특정 스캔 규칙을 사용하며, 인과적 특성으로 인한 정보 손실을 막기 위해 4방향 스캔 등의 다방향 스캔 방식을 사용한다. 하지만 본 논문은 이러한 방식이 계산 비용을 높이고 중복 정보를 생성한다는 점을 지적하며, 구조적으로 비인과적 특성을 부여하는 접근 방식과의 차별성을 강조한다.

## 🛠️ Methodology

### 1. Attention과 State Space의 연결 분석
저자들은 선형 어텐션(Linear Attention)과 상태 공간 방정식의 형태를 재구성하여 수학적 유사성을 분석하였다. 분석 결과, 상태 공간 방정식의 출력 행렬 $C$가 어텐션 메커니즘의 쿼리(Query)와 유사한 역할을 수행함을 발견하였다. 이 통찰을 바탕으로 $C$ 행렬에 전역 정보를 담은 프롬프트를 주입함으로써 Mamba가 어텐션처럼 전역 픽셀을 쿼리하도록 설계하였다.

### 2. Attentive State-space Equation (ASE)
ASE는 Mamba의 기본 방정식에 학습 가능한 프롬프트를 추가하여 비인과성을 구현한다.

- **프롬프트 풀(Prompt Pool) $P$**: 의미적 해독(Semantic Decoupling)을 위해 $P = MN$ 형태로 정의한다. 여기서 $N \in \mathbb{R}^{r \times d}$는 모든 블록이 공유하는 공통 특징 공간이며, $M \in \mathbb{R}^{T \times r}$은 블록별로 특화된 결합 계수이다.
- **라우팅 전략**: 입력 특징 $x'$를 선형 층과 $\text{LogSoftmax}$를 거쳐 각 프롬프트가 선택될 확률을 계산하고, $\text{Gumbel-Softmax}$ 트릭을 통해 미분 가능한 원-핫(one-hot) 라우팅 행렬 $R$을 생성한다. 이를 통해 인스턴스별 프롬프트 $\mathcal{P} = RP$를 얻는다.
- **수정된 상태 공간 방정식**:
  $$h_i = Ah_{i-1} + Bx_i$$
  $$y_i = (C + \mathcal{P})h_i + Dx_i$$
  기존의 출력 행렬 $C$에 생성된 프롬프트 $\mathcal{P}$를 잔차 연결(Residual Addition) 방식으로 더함으로써, 쿼리 픽셀이 스캔 시퀀스 너머의 유사한 픽셀 정보를 참조할 수 있게 한다.

### 3. Semantic Guided Neighboring (SGN)
Mamba의 장거리 감쇠 문제를 해결하기 위해, 픽셀의 배치 순서를 변경하는 SGN을 도입한다.

- **SGN-unfold**: ASE의 라우팅 행렬 $R$에서 얻은 의미적 레이블(Semantic Label)을 기준으로, 동일한 프롬프트 카테고리에 속하는 픽셀들을 그룹화하여 1D 시퀀스를 재구성한다. 즉, 이미지 상에서 멀리 떨어져 있더라도 의미적으로 유사한 픽셀들이 시퀀스 상에서는 서로 가깝게 위치하게 된다.
- **SGN-fold**: ASE 모델링이 끝난 후, 재구성된 시퀀스를 다시 원래의 2D 이미지 공간 좌표로 되돌리는 역변환을 수행한다.

### 4. 전체 네트워크 구조
MambaIRv2는 국소적-전역적 단계적 모델링(Progressive Local-to-Global Modeling) 구조를 따른다.
- **구조**: 얕은 특징 추출 $\rightarrow$ Attentive State Space Groups (ASSGs) $\rightarrow$ 재구성 모듈(PixelShuffle 등).
- **ASSB (Block) 구성**: 각 블록은 $\text{Norm} \rightarrow \text{Token Mixer} \rightarrow \text{Norm} \rightarrow \text{FFN}$ 구조를 가진다. 이때 $\text{Token Mixer}$는 국소적 상호작용을 위한 **Window MHSA**와 전역적 상호작용을 위한 **ASSM(ASE + SGN)**으로 구성되어 이미지의 계층적 구조를 학습한다.

## 📊 Results

### 1. 실험 설정
- **작업**: 고전적 초해상도(Classic SR), 경량 초해상도(Lightweight SR), JPEG 압축 아티팩트 제거(JPEG CAR), 가우시안 컬러 이미지 디노이징(Denoising).
- **데이터셋**: DIV2K, DF2K, Set5, Set14, B100, Urban100, Manga109 등.
- **지표**: PSNR, SSIM, 파라미터 수, MACs.

### 2. 주요 결과
- **경량 SR (Lightweight SR)**: MambaIRv2-light는 2$\times$ Urban100 데이터셋에서 SRFormer-light보다 파라미터 수를 9.3% 줄였음에도 불구하고 PSNR을 0.35dB 높였다.
- **고전적 SR (Classic SR)**: MambaIRv2-B/L 모델은 대부분의 벤치마크에서 SOTA 성능을 달성하였다. 특히 2$\times$ Manga109에서 HAT보다 0.29dB 높은 성능을 보였다.
- **효율성**: 기존 MambaIR이 4방향 스캔을 사용한 것과 달리, MambaIRv2는 단일 방향 스캔만으로 더 높은 성능을 냈으며, MACs(연산량) 면에서도 HAT 대비 약 13.4% 감소(MambaIRv2-B 기준)하는 효율성을 입증하였다.
- **일반화 성능**: JPEG CAR 및 Denoising 작업에서도 기존 MambaIR 및 Restormer 등의 모델보다 우수한 성능을 기록하여, 범용적인 복원 백본으로서의 가능성을 보여주었다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **비인과적 모델링의 실효성**: 본 논문은 Mamba의 결정적 약점인 인과적 제약을 프롬프트 기반의 ASE를 통해 성공적으로 극복하였다. 이는 다방향 스캔이라는 무거운 우회책 없이도 전역 수용 영역을 확보할 수 있음을 시사한다.
- **의미적 재배치의 효과**: SGN을 통해 물리적 거리가 아닌 의미적 거리 기반으로 시퀀스를 재구성함으로써, Mamba의 고유한 문제인 장거리 감쇠(Long-range Decay)를 효과적으로 억제하였다.
- **전역 지각 능력 검증**: LAM(Local Attribution Map) 및 ERF(Effective Receptive Field) 시각화 결과, MambaIRv2가 타 모델 대비 훨씬 넓고 균일한 전역 수용 영역을 가짐을 확인하였다. 특히 기존 MambaIR에서 나타나던 특유의 'X자형' ERF(인과적 스캔의 흔적)가 사라진 점은 비인과적 모델링이 성공적으로 적용되었음을 증명한다.

### 2. 한계 및 논의사항
- **프롬프트 풀 크기의 민감도**: 보충 자료에서 프롬프트 풀의 크기 $T$와 내부 랭크 $r$에 따른 성능 변화가 관찰되었다. 특정 임계값을 넘으면 오히려 성능이 소폭 하락하는 경향이 있어, 적절한 하이퍼파라미터 튜닝이 필수적이다.
- **해석 가능성의 필요성**: 저자들은 Mamba가 복원 과정에서 구체적으로 무엇을 학습하는지에 대한 심층적인 해석 분석이 향후 연구 과제임을 명시하였다.
- **적용 범위의 확장**: 현재 SR, Denoising 등에 적용되었으나, Deblurring, Dehazing 등 더 다양한 저수준 비전 작업으로의 확장이 필요하다.

## 📌 TL;DR

본 논문은 Mamba의 인과적 모델링 제약으로 인해 발생하는 전역 정보 활용 저하, 계산 중복, 장거리 감쇠 문제를 해결하기 위해 **MambaIRv2**를 제안한다. **ASE(Attentive State-space Equation)**를 통해 전역 프롬프트를 주입하여 비인과적 쿼리를 가능케 함으로써 단일 스캔만으로 전역 수용 영역을 확보하였고, **SGN(Semantic Guided Neighboring)**을 통해 의미적으로 유사한 픽셀들을 가깝게 배치하여 장거리 상호작용을 강화하였다. 실험 결과, MambaIRv2는 ViT 기반의 SOTA 모델(HAT, SRFormer 등)보다 적은 파라미터와 연산량으로도 더 높은 복원 성능을 달성하며, 효율성과 효과성을 동시에 잡은 차세대 이미지 복원 백본임을 입증하였다.