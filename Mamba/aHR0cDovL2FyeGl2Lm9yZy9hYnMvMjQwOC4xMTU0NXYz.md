# UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images

Enze Zhu, Zhan Chen, Dingkai Wang, Hanru Shi, Xiaoxuan Liu, and Lei Wang (2024)

## 🧩 Problem to Solve

고해상도 원격 탐사 이미지(High-Resolution Remote Sensing Images)의 시맨틱 세그멘테이션(Semantic Segmentation)은 토지 피복 지도 작성, 도시 계획, 재난 평가 등 다양한 하위 응용 분야에서 필수적이다. 그러나 고해상도 이미지는 정보의 복잡성이 매우 높기 때문에, 이를 처리하는 모델은 높은 정확도와 효율성이라는 두 가지 상충하는 목표를 동시에 달성해야 하는 어려움이 있다.

기존의 Transformer 기반 방법론들은 전역적 문맥(Global Context)을 파악하여 높은 정확도를 제공하지만, 계산 복잡도가 입력 크기의 제곱에 비례하는 Quadratic Computational Complexity 문제로 인해 고해상도 이미지 처리 시 효율성이 크게 떨어진다. 반면, CNN 기반 방법론은 효율적이지만 전역적인 의존성을 모델링하는 능력이 부족하다. 따라서 본 논문의 목표는 Mamba 아키텍처의 선형 복잡도(Linear Complexity)와 강력한 장거리 의존성 모델링 능력을 활용하여, 고해상도 원격 탐사 이미지 세그멘테이션에서 정확도와 효율성의 딜레마를 해결하는 UNetMamba 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 효율적인 전역 모델링 능력을 디코더에 배치하고, Mamba가 놓치기 쉬운 국소적 세부 정보를 보완하기 위해 훈련 단계에서만 작동하는 가벼운 CNN 기반 감독 모듈을 추가하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Mamba Segmentation Decoder (MSD) 설계**: VMamba의 Visual State Space (VSS) 블록을 디코더 측에 적용하여, 파라미터 수를 줄이면서도 전역 수용 영역(Global Receptive Field) 내에서 복잡한 정보를 효율적으로 디코딩할 수 있는 플러그 앤 플레이 방식의 디코더를 구현하였다.
2. **Local Supervision Module (LSM) 도입**: Mamba의 넓은 수용 영역으로 인해 발생할 수 있는 국소적 세부 정보 손실 문제를 해결하기 위해 CNN 기반의 LSM을 제안하였다. 이는 훈련 단계에서만 사용되는 Train-only 설계로, 추론 시의 비용 증가 없이 국소적 시맨틱 정보에 대한 인지 능력을 향상시킨다.
3. **UNetMamba 프레임워크 구축**: ResT 백본의 인코더, MSD, LSM을 결합한 UNet 형태의 구조를 통해 경량화와 낮은 계산 비용을 유지하면서도, 주요 원격 탐사 데이터셋에서 SOTA(State-of-the-Art) 성능을 달성하였다.

## 📎 Related Works

기존의 원격 탐사 이미지 세그멘테이션 연구는 크게 CNN 기반과 Transformer 기반으로 나뉜다. UNet은 인코더-디코더 구조와 스킵 연결(Skip Connection)을 통해 기초적인 아키텍처를 제시하였으며, 이후 Transformer가 도입되면서 정확도가 크게 향상되었다. 하지만 Transformer 기반 모델들은 높은 파라미터 수와 계산 복잡도로 인해 고해상도 이미지 적용에 한계가 있었다.

최근 등장한 Mamba는 선형 복잡도를 가지면서도 장거리 의존성을 효과적으로 모델링할 수 있어 효율적인 대안으로 주목받고 있다. 이미 의료 영상 세그멘테이션이나 원격 탐사 이미지의 분류($RSMamba$), 변화 탐지($CDMamba$) 등에 적용되어 가능성을 보여주었다. 그러나 기존의 Mamba 기반 세그멘테이션 연구들은 단순히 Mamba를 적용하는 데 그쳤을 뿐, 정확도와 효율성을 동시에 극대화하기 위한 타겟팅된 설계(Targeted Design)가 부족했다는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

UNetMamba는 U-shape 프레임워크를 따르며, 크게 세 가지 구성 요소로 이루어져 있다.

- **Encoder**: 사전 학습된 ResT 백본을 사용하여 다중 스케일 특징 맵을 추출한다.
- **Decoder (MSD)**: Mamba 기반의 디코더로, 인코더에서 추출된 특징을 효율적으로 복원하고 전역적 문맥을 반영한다.
- **Local Supervision Module (LSM)**: 훈련 과정에서 MSD의 국소적 인지 능력을 강화하는 보조 모듈이다.

### 2. ResT Encoder

ResT 인코더는 Efficient Transformer Block (ETB)을 핵심으로 하며 4단계의 스테이지로 구성된다. 계산 비용을 줄이기 위해 Efficient Multi-head Self-Attention (EMSA)를 사용하며, 수식은 다음과 같다.

$$\text{EMSA}(Q, K, V) = \text{IN}\left(\text{Softmax}\left(\frac{\text{Conv}(QK^T)}{\sqrt{d_k}}\right)V\right)$$

여기서 $\text{IN}(\cdot)$은 Instance Normalization이며, $1 \times 1$ Convolution을 도입하여 효율성을 높였다.

### 3. Mamba Segmentation Decoder (MSD)

MSD는 VMamba의 기본 단위인 VSS 블록을 사용한다. 입력 특징 맵 $F$는 Patch Expansion을 거쳐 다음과 같은 과정을 수행한다.

- **선형 임베딩**: $F' = \text{LinearEmbed}(\text{LayerNorm}(\text{PatchExp}(F)))$
- **특징 추출**: $F'' = \text{SiLU}(\text{DWConv}(F'))$ (Depth-wise Convolution 및 SiLU 활성화 함수 적용)
- **2-D Selective Scan (SS2D)**: $F''$를 네 가지 방향으로 스캔하여 전역 수용 영역의 정보를 선형 복잡도로 디코딩한다.
  $$F_v = \text{ScanExp}(F'', v), \quad v \in \{1, 2, 3, 4\}$$
  $$\bar{F}_v = \text{S6}(F_v), \quad v \in \{1, 2, 3, 4\}$$
  $$\bar{F} = \text{ScanMerge}(\bar{F}_1, \bar{F}_2, \bar{F}_3, \bar{F}_4)$$
  여기서 $\text{S6}$는 Mamba의 Selective State Space 모델을 의미한다.

최종 출력 $F_{\text{MSD}}$는 잔차 연결(Residual Connection)과 Linear 레이어를 통해 생성되며, 4단계의 스테이지를 거쳐 최종적으로 $1 \times 1$ Convolution 헤드를 통해 세그멘테이션 결과를 출력한다.

### 4. Local Supervision Module (LSM)

MSD의 넓은 수용 영역이 국소적 세부 정보를 간과하는 문제를 해결하기 위해 CNN 기반의 LSM이 도입되었다.

- **병렬 컨볼루션**: 커널 크기가 1과 3인 두 개의 병렬 브랜치를 사용하며, 각각 $\text{BatchNorm}$과 $\text{ReLU6}$를 거친다.
  $$\tilde{F}'_i = \text{ReLU6}(\text{BatchNorm}(\text{Conv}_i(F_{\text{MSD}}))), \quad i \in \{1, 3\}$$
- **결합 및 업샘플링**: 두 브랜치를 합산($\tilde{F}' = \tilde{F}'_1 \oplus \tilde{F}'_3$)한 후, $\text{Dropout} \rightarrow 1 \times 1 \text{Conv} \rightarrow \text{Upsample}$ 과정을 거쳐 최종 결과 $\tilde{F}$를 얻는다.
이 모듈은 훈련 시에만 디코더의 스테이지 2~4에 추가되어 보조 손실 함수를 계산하는 데 사용된다.

### 5. Loss Function

전체 손실 함수 $L$은 주 손실(Principal Loss) $L_p$와 보조 손실(Auxiliary Loss) $L_a$의 가중 합으로 정의된다.

$$L = L_p + \alpha L_a = (L_{\text{dice}} + L_{\text{ce}}) + \alpha L_{\text{ce}}$$

여기서 $\alpha = 0.4$이며, $L_{\text{dice}}$는 Dice Loss, $L_{\text{ce}}$는 Cross-Entropy Loss이다. 주 손실은 전체적인 최적화를 담당하고, 보조 손실은 LSM을 통한 국소적 감독을 담당한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: LoveDA (5,987장, 7개 클래스), ISPRS Vaihingen (33장, 6개 클래스).
- **평가 지표**: mIoU, mF1, OA (정확도), Param, Memo, FLOPs (효율성).
- **환경**: NVIDIA RTX 4090 GPU, AdamW 옵티마이저.

### 2. 정량적 결과

- **LoveDA**: UNetMamba는 mIoU $53.35\%$를 기록하며 기존 SOTA 대비 $0.87\%$ 향상되었다. 특히 배경(Background)과 농경지(Agriculture) 클래스에서 각각 $1.48\%$, $2.39\%$의 큰 격차로 우위를 점했다.
- **ISPRS Vaihingen**: mF1 $90.95\%$, mIoU $83.47\%$, OA $92.51\%$를 달성하여 기존 모델들을 앞섰다. 특히 모델 파라미터 수($14.76\text{M}$)와 계산 비용($100.52\text{G FLOPs}$) 측면에서 매우 경쟁력 있는 효율성을 보여주었다.

### 3. 절제 연구 (Ablation Study)

- **MSD의 효과**: ResT-Lite 백본을 기반으로 MSD를 적용했을 때, 파라미터 수와 FLOPs가 크게 감소하면서도 mIoU 하락은 매우 적었다. 이는 MSD가 복잡한 정보를 매우 효율적으로 디코딩함을 증명한다.
- **LSM의 효과**: LSM을 추가했을 때 LoveDA에서는 $0.74\%$, Vaihingen에서는 $0.29\%$의 mIoU 상승이 나타났다. 파라미터 증가는 $0.87\text{M}$으로 매우 적었으며, 추론 시에는 사용되지 않으므로 실질적인 비용 증가가 없었다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 선형 복잡도라는 강점을 디코더에 전략적으로 배치함으로써 고해상도 이미지 처리의 효율성 문제를 해결하였다. 특히 주목할 점은 **"전역적 모델링(Mamba)과 국소적 보완(CNN)"**의 조화이다. Mamba의 넓은 수용 영역이 가져오는 '국소 정보 소실'이라는 부작용을 LSM이라는 가벼운 CNN 구조와 Train-only 전략으로 해결한 점은 매우 영리한 설계라고 판단된다.

또한, 무거운 Mamba 백본을 그대로 사용하는 대신, 인코더는 효율적인 ResT를 사용하고 디코더에만 Mamba 블록을 배치함으로써 파라미터 수를 획기적으로 줄이면서도 SOTA 성능을 낸 점이 인상적이다. 다만, 논문에서 제시된 $\alpha=0.4$라는 가중치 설정의 근거가 명확히 제시되지 않았으며, 다른 하이퍼파라미터에 대한 민감도 분석이 부족한 점은 아쉬운 부분이다.

## 📌 TL;DR

**UNetMamba**는 고해상도 원격 탐사 이미지의 시맨틱 세그멘테이션을 위해 제안된 효율적인 모델이다. **Mamba 기반의 디코더(MSD)**를 통해 전역적 문맥을 선형 복잡도로 처리하고, **훈련 전용 국소 감독 모듈(LSM)**을 통해 세부 디테일을 보완하였다. 그 결과, 매우 가벼운 모델 구조(약 $14.76\text{M}$ 파라미터)임에도 불구하고 LoveDA 및 ISPRS Vaihingen 데이터셋에서 기존 Transformer 및 Mamba 기반 모델들보다 뛰어난 정확도와 효율성을 입증하였다. 이 연구는 향후 고해상도 영상 분석 분야에서 Mamba 아키텍처를 효율적으로 설계하는 가이드라인을 제시할 것으로 기대된다.
