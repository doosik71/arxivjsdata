# A Hybrid Transformer-Mamba Network for Single Image Deraining

Shangquan Sun, Wenqi Ren, Juxiang Zhou, Jianhou Gan, Rui Wang, Xiaochun Cao (2021/2024)

## 🧩 Problem to Solve

본 논문은 단일 이미지 제우(Single Image Deraining) 작업에서 발생하는 비국소적 수용 영역(non-local receptive fields) 확보의 어려움을 해결하고자 한다. 기존의 CNN 기반 방법론은 합성곱 층의 제한된 수용 영역으로 인해 장거리 의존성(long-range dependencies)을 학습하는 데 한계가 있었다. 이를 극복하기 위해 등장한 Vision Transformer 기반 방법론들은 계산 복잡도 문제로 인해 고정된 크기의 윈도우(fixed-range windows) 내에서 혹은 채널 차원을 따라 Self-attention을 수행한다. 이러한 방식은 이미지 전체에 동적으로 분포된 빗줄기(rain streaks) 간의 상관관계를 충분히 활용하지 못한다는 문제점이 있다. 따라서 본 연구의 목표는 장거리 의존성을 효과적으로 포착하여 다양한 스케일과 형태의 빗줄기를 제거할 수 있는 하이브리드 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer와 Mamba(State Space Model)의 장점을 결합한 dual-branch 하이브리드 네트워크인 **TransMamba**를 설계하는 것이다.

1. **Spectral-Domain Transformer Branch**: 빗줄기가 스펙트럼 영역(spectral domain)에서 특정 저주파 성분에 집중된다는 특성을 이용하여, 주파수 밴드별로 주의(attention)를 다르게 할당하는 Spectral Banding Self-Attention (SBSA)을 제안한다. 이를 통해 저주파의 빗줄기 신호는 억제하고 고주파의 배경 세부 사항은 보존한다.
2. **Mamba Branch**: 시퀀스 일관성(sequence coherence)을 강화하기 위해 Cascaded Bidirectional State Space Model (CBSM) 모듈을 도입하여 국소적 정보와 전역적 정보를 동시에 포착한다.
3. **Spectral Enhanced Feed-Forward (SEFF)**: 공간 도메인의 합성곱이 주파수 도메인의 요소별 곱셈과 동일하다는 성질을 이용하여, 적응형 주파수 통과 필터 역할을 하는 SEFF 모듈을 통해 주파수 특화 정보를 강화한다.
4. **Spectral Coherence Loss**: 픽셀 단위의 유사도를 넘어 신호 수준의 선형 관계를 복원하기 위해 스펙트럼 일관성 손실 함수를 도입하여 결과물의 일관성을 높인다.

## 📎 Related Works

기존의 제우 연구는 크게 물리적 사전 지식(Physical Priors) 기반 방법, CNN 기반 방법, 그리고 Transformer 기반 방법으로 발전해 왔다.

- **사전 지식 기반 방법**: Sparse coding, Low-rank model 등을 사용했으나 하이퍼파라미터 튜닝에 의존적이며 복잡한 빗줄기 제거에 한계가 있다.
- **CNN 기반 방법**: 복잡한 패턴 학습 능력이 뛰어나지만 수용 영역의 제한으로 인해 전역적인 의존성을 학습하기 어렵다.
- **Transformer 기반 방법**: Self-attention을 통해 비국소적 표현을 학습하지만, 계산 복잡도로 인해 윈도우 기반이나 채널 기반의 제한적인 attention을 사용하며, 이는 빗줄기의 전역적 상관관계를 무시하는 결과를 초래한다.
- **Mamba/SSM 기반 방법**: 최근 효율적인 장거리 모델링 능력이 입증되었으나, 이미지 복원 작업에서는 국소적 정보 추출 능력이 부족할 수 있다는 지적이 있다.

본 논문은 이러한 한계를 극복하기 위해 Transformer의 전역적 모델링 능력과 Mamba의 시퀀스 일관성 모델링 능력을 하이브리드 구조로 결합하고, 이를 주파수 도메인(Spectral Domain)에서 수행함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 전체 아키텍처

TransMamba는 대칭적인 다단계 dual-branch 인코더-디코더 구조를 가진다. 입력 이미지 $R \in \mathbb{R}^{3 \times H \times W}$는 패치 임베딩을 거쳐 두 개의 브랜치로 나뉜다.

- **Branch 1**: Spectral-Domain Transformer Blocks (SDTBs)를 통해 전역적 의존성과 주파수 특성을 추출한다.
- **Branch 2**: Cascaded Bidirectional SSM Modules (CBSMs)를 통해 시퀀스 일관성을 보완한다.
각 단계에서 두 브랜치의 특징 맵은 채널 방향으로 결합(concatenation)된 후 pointwise convolution을 통해 융합되며, Skip-connection을 통해 학습 안정성을 높인다.

### Spectral-Domain Transformer Block (SDTB)

SDTB는 다음의 두 단계로 구성된다.
$$F_n = F_{n-1} + \text{SBSA}[\text{LayerNorm}(F_{n-1})]$$
$$F_n = F_n + \text{SEFF}[\text{LayerNorm}(F_n)]$$

#### 1) Spectral Banding Self-Attention (SBSA)

SBSA는 공간 도메인의 특징을 2D FFT를 통해 주파수 도메인으로 변환한 후 attention을 수행한다.

- **SBR (Spectral Banding Reorganization)**: 주파수 성분을 고주파에서 저주파 순으로 정렬한 뒤 여러 개의 밴드(band)로 나눈다. 빗줄기가 포함된 저주파 밴드에는 낮은 attention을, 배경 정보가 담긴 고주파 밴드에는 높은 attention을 할당하여 효과적으로 빗줄기를 제거한다.
- 이후 IFFT를 통해 다시 공간 도메인으로 복원한다.

#### 2) Spectral Enhanced Feed-Forward (SEFF)

SEFF는 다중 범위(multi-range) 정보를 추출하기 위해 $3 \times 3$ depth-wise convolution과 dilated $3 \times 3$ depth-wise convolution을 병렬로 사용한다.

- 각 경로의 특징은 FFT를 통해 주파수 도메인으로 변환되며, 학습 가능한 가중치 $W$와 편향 $B$를 이용해 요소별 곱셈($\odot$)을 수행함으로써 적응형 필터링을 수행한다.
- 한 브랜치의 출력이 SiLU 활성화 함수를 통해 다른 브랜치의 게이팅(gating) 유닛으로 작동하여 최적의 스펙트럼 강화를 달성한다.

### Cascaded Bidirectional SSM Modules (CBSM)

Mamba 브랜치는 SSM의 망각 문제를 해결하기 위해 정방향(forward)과 역방향(backward) SSM을 캐스케이드 구조로 배치한다. 정방향 처리 후 채널을 뒤집어(flip) 역방향 처리를 수행함으로써 이미지 시퀀스의 선형 의존성을 보존하고 전역 정보를 보강한다.

### 손실 함수 (Loss Functions)

모델은 재구성 손실(Reconstruction Loss)과 스펙트럼 일관성 손실(Spectral Coherence Loss)의 합으로 학습된다.

- **Reconstruction Loss**: 제우 이미지 $\tilde{B}$와 깨끗한 이미지 $B$ 사이의 $L1$ norm을 사용한다.
$$\mathcal{L}_{rec} = \|\tilde{B} - B\|_1$$
- **Spectral Coherence Loss**: 두 신호 간의 선형 관계를 측정하는 coherence $G(\tilde{B}, B)$를 정의하여 신호 수준의 복원을 유도한다.
$$\mathcal{L}_{coh} = 1 - \sqrt{G(\tilde{B}, B)}$$
- **최종 손실 함수**: $\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{coh}$ (여기서 $\alpha$는 가중치이다).

## 📊 Results

### 실험 설정

- **데이터셋**: 합성 데이터셋(Rain200H, Rain200L, DID-Data, DDN-Data)과 실제 데이터셋(SPA-Data, Internet-Data)을 사용하였다.
- **비교 대상**: DSC, GMM 등의 Prior 기반 모델, PReNet, MPRNet 등의 CNN 모델, Restormer, IDT, NeRD-Rain 등의 Transformer 모델 및 DFSSM 등 Mamba 기반 모델과 비교하였다.
- **평가 지표**: PSNR, SSIM 및 무참조 품질 평가 지표인 NIQE, BRISQUE를 사용하였다.

### 정량적 및 정성적 결과

- **정량적 성능**: 모든 데이터셋에서 SOTA 성능을 달성하였다. 특히 기존 최우수 모델인 UDR-S2Former 대비 평균 0.32 dB의 PSNR 향상을 보였다.
- **실제 이미지 품질**: Internet-Data에서 NIQE(3.31)와 BRISQUE(18.51) 지표 모두 최저치를 기록하며 가장 시각적으로 우수한 결과물을 생성하였다.
- **효율성**: 파라미터 수와 FLOPs 측면에서 기존 SOTA 모델들과 유사한 수준의 복잡도를 유지하면서 성능을 높였다.
- **다운스트림 작업**: 제우된 이미지를 YOLOv3 객체 검출기에 입력한 결과, 빗줄기로 인해 가려졌던 물체(신발, 핸드백 등)의 검출 정밀도(Precision)와 재현율(Recall)이 유의미하게 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 이미지의 주파수 특성을 이용하여 빗줄기와 배경을 분리하고, Transformer의 전역 모델링 능력과 Mamba의 시퀀스 일관성을 결합함으로써 단일 이미지 제우 성능을 극대화하였다. 특히 주파수 도메인에서 attention을 수행하는 SBSA와 신호 수준의 관계를 보존하는 Spectral Coherence Loss의 도입이 성능 향상에 결정적인 역할을 한 것으로 분석된다.

**한계점 및 비판적 해석**:

- 논문에서 언급했듯이, 강력한 빗줄기 제거 능력으로 인해 복원된 배경이 과도하게 매끄러워지는(over-smooth) 경향이 있다. 이는 고주파 성분을 보존하려는 노력에도 불구하고 일부 텍스처 손실이 발생함을 의미한다.
- 제안된 모델의 복잡도가 기존 모델과 유사하다고 주장하지만, FFT 및 IFFT 연산이 반복적으로 수행되므로 실제 추론 속도(Runtime)에 미치는 영향에 대한 더 상세한 분석이 필요하다. (다만, 표 III에서는 Runtime이 매우 경쟁력 있음을 보여준다.)

## 📌 TL;DR

본 연구는 Transformer의 전역적 특성과 Mamba의 시퀀스 모델링 능력을 결합한 **TransMamba** 네트워크를 제안하였다. 주파수 도메인에서 밴드별 주의 집중을 수행하는 SBSA와 적응형 필터링을 수행하는 SEFF, 그리고 bidirectional SSM을 통해 빗줄기를 효과적으로 제거하며, 스펙트럼 일관성 손실 함수를 통해 신호의 원형을 복원한다. 실험적으로 SOTA 성능을 입증하였으며, 특히 제우 후 객체 검출(Object Detection) 성능이 향상됨으로써 실제 비전 파이프라인에서의 유용성을 증명하였다.
