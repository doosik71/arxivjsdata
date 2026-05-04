# Mamba Based Feature Extraction And Adaptive Multilevel Feature Fusion For 3D Tumor Segmentation From Multi-modal Medical Image

Zexin Ji, Beiji Zou, Xiaoyan Kui, Hua Li, Pierre Vera, and Su Ruan (2025)

## 🧩 Problem to Solve

본 연구는 다중 모달(multi-modal) 3D 의료 영상에서 종양 영역을 정확하게 식별하는 3D tumor segmentation 문제를 해결하고자 한다. 의료 영상의 각 모달리티(modality)는 서로 다른 강도(intensity)와 종양 형태의 특성을 가지고 있어 이를 통합적으로 분석하는 것이 매우 중요하다.

기존의 접근 방식들은 다음과 같은 한계점을 가지고 있다. 첫째, CNN 기반 방법론은 수용 영역(receptive field)이 국소적이기 때문에 전역적 특징(global features)을 포착하는 데 어려움이 있다. 둘째, Transformer 기반 방법론은 전역적 문맥 파악에는 유리하나, 3D 의료 영상과 같은 대용량 데이터 처리 시 계산 복잡도가 시퀀스 길이의 제곱에 비례하여 계산 비용이 매우 높다. 셋째, 다중 모달 데이터를 단순히 채널 방향으로 쌓아서 입력하는 방식은 각 모달리티의 고유한 특성과 상호작용을 명시적으로 모델링하지 못하며, 고정된 융합 전략을 사용하여 동적인 특징 조절이 어렵다.

따라서 본 논문의 목표는 Mamba 아키텍처를 활용하여 계산 효율성을 유지하면서 전역적 문맥을 포착하고, 모달리티별 특성을 보존하며 적응적으로 융합하는 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 각 모달리티의 독립적인 특징 추출과 적응형 다층 융합을 결합하는 것이다.

1. **Specific Modality Mamba Encoder**: 각 모달리티를 독립적으로 처리하는 인코더를 설계하여, 융합 전 각 영상이 가진 해부학적 및 병리학적 구조의 고유한 특징을 효율적으로 추출하고 모달리티 간의 간섭이나 노이즈 영향을 최소화한다.
2. **Bi-level Synergistic Integration Block**: 모달리티 수준(Modality-level)과 채널 수준(Channel-level)의 어텐션을 동시에 적용하는 이중 레벨 융합 블록을 통해, 각 모달리티의 중요도와 내부 채널의 유용성을 동적으로 조절하여 상호 보완적인 정보를 최적으로 통합한다.
3. **Mamba-based Decoder Enhancement**: 디코더의 병목(bottleneck) 지점에 Mamba Block을 추가하여, 압축 과정에서 손실될 수 있는 공간 정보를 선택적으로 증폭시키고 세밀한 세그멘테이션 맵을 생성한다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 흐름과 한계를 언급한다.

- **CNN 기반 방법**: $\text{BraTS 2018}$ 챌린지 우승 모델과 같이 Encoder-Decoder 구조를 활용한 연구들이 많으나, 국소적 수용 영역으로 인해 장거리 의존성(long-range dependencies)을 학습하는 능력이 부족하다.
- **Transformer 기반 방법**: $\text{Swin UNETR}$과 같이 Self-attention 메커니즘을 통해 전역 문맥을 포착하는 능력이 뛰어나지만, 3D 데이터의 긴 시퀀스 길이에 따른 높은 계산 비용이 문제로 지적된다.
- **Mamba 기반 방법**: 최근 등장한 State Space Model(SSM)인 Mamba는 시퀀스 길이에 대해 선형적인 확장성(linear scalability)을 가지면서도 전역 모델링이 가능하여 $\text{SegMamba}$와 같은 3D 의료 영상 분할 모델로 확장되고 있다. 하지만 여전히 다중 모달리티 간의 보완적 정보를 효과적으로 융합하는 메커니즘은 부족한 실정이다.

## 🛠️ Methodology

### 1. Specific Modality Mamba Encoder

각 모달리티는 서로 다른 특성(예: T1-Gd는 강화된 종양 영역, FLAIR는 부종에 민감)을 가지므로, 개별 Mamba 인코더를 통해 특징을 추출한다. Mamba의 핵심인 상태 공간 모델(SSM)은 다음과 같은 연속 시간 방정식으로 정의된다.

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

여기서 $A$는 상태 행렬, $B$와 $C$는 투영 파라미터이다. 이를 딥러닝에 적용하기 위해 Zero-Order Hold (ZOH) 방식을 사용하여 이산화(discretization)하며, 타임스케일 파라미터 $\Delta$를 통해 이산 형태의 $\bar{A}, \bar{B}$로 변환한다.

$$ \bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B $$

최종적으로 이산화된 모델은 다음과 같이 작동한다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad y_t = C h_t$$

본 모델의 Mamba Block은 $\text{Layer Norm} \rightarrow \text{Linear} \rightarrow \text{DW Conv} \rightarrow \text{SS3D (3D-Selective-Scan)} \rightarrow \text{Layer Norm} \rightarrow \text{Linear}$ 경로와 단순 선형 경로를 병렬로 구성하며, SS3D를 통해 전방향 및 후방향의 문맥 정보를 모두 활용한다. 이후 $\text{Res Block}$을 추가하여 세밀한 공간 세부 사항을 보완한다.

### 2. Bi-Level Synergistic Integration Block

다양한 모달리티에서 추출된 특징 맵 $\mathcal{X} = \{X^{(1)}, X^{(2)}, \dots, X^{(M)}\}$ (여기서 $X^{(m)} \in \mathbb{R}^{C \times H \times W \times D}$)을 입력으로 받는다.

- **Modality-level Attention**: 모든 모달리티 특징을 결합(concatenation)하고 $\text{AvgPool}$을 적용하여 전역 기술자를 생성한 후, $\text{Linear} \rightarrow \text{ReLU} \rightarrow \text{Linear} \rightarrow \text{Softmax}$ 과정을 통해 각 모달리티의 가중치 $A_{modality}$를 계산한다.
$$A_{modality} = \text{Softmax}(W_{mod2} \cdot \sigma(W_{mod1} \cdot X_{pool}))$$
- **Channel-level Attention**: 동일한 $X_{pool}$을 사용하여 $\text{Linear} \rightarrow \text{ReLU} \rightarrow \text{Linear} \rightarrow \text{Sigmoid}$ 과정을 통해 채널별 가중치 $A_{channel}$을 계산한다.
$$A_{channel} = \sigma(W_{ch2} \cdot \sigma(W_{ch1} \cdot X_{pool}))$$
- **Final Integration**: 계산된 가중치를 이용하여 다음과 같이 특징을 재조정한다.
$$X^{(m)}_{out} = A^{(m)}_{modality} \cdot (A_{channel} \odot X^{(m)})$$
여기서 $\odot$은 채널 차원에서의 요소별 곱셈을 의미한다.

### 3. Decoder

디코더는 업샘플링 층과 인코더로부터의 skip connection을 통해 공간 해상도를 점진적으로 복원한다. 특히 bottleneck 지점에 Mamba Block을 배치하여 압축된 특징에서 중요한 정보를 선택적으로 증폭시킨다. 최종적으로 $\text{Res block}$을 통해 고수준의 시맨틱 정보와 저수준의 공간 세부 정보를 결합하여 종양 세그멘테이션 맵을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: $\text{BraTS2023}$ (MRI: T1, T1-Gd, T2, FLAIR) 및 $\text{Hecktor2022}$ (PET, CT).
- **평가 지표**: $\text{Dice Score}$ (예측 영역과 실제 영역의 겹침 정도), $\text{Hausdorff Distance}$ (경계면의 최대 거리 편차).
- **구현**: PyTorch, NVIDIA RTX A6000, SGD 옵티마이저, Cross-entropy 손실 함수 사용.

### 주요 결과

1. **Ablation Study (BraTS2023)**:
   - $\text{SingleModality} < \text{SimpleFusion} < \text{MambaEncoder} < \text{Ours}$ 순으로 성능이 향상되었다.
   - 특히 $\text{MambaEncoder}$에 Bi-level synergistic integration block을 추가한 제안 방법이 Dice Score에서 2.02% 상승, Hausdorff Distance에서 1.16mm 감소하는 성과를 보였다.
2. **SOTA 비교 (BraTS2023)**:
   - CNN 기반 ($\text{UX-Net}$, $\text{MedNeXt}$), Transformer 기반 ($\text{SwinUNETR-V2}$), Mamba 기반 ($\text{SegMamba}$) 모든 모델보다 우수한 성능을 기록했다.
   - 특히 경계가 불분명하고 크기가 작은 $\text{Enhancing Tumor (ET)}$ 영역에서 더 높은 Dice Score와 낮은 HD를 기록하여 세밀한 경계 묘사 능력을 입증했다.
3. **Hecktor2022 결과**:
   - $\text{PET-only}$ 방식보다 $\text{PET/CT}$ 융합 방식(제안 방법)이 더 우수한 성능을 보였다. 이는 PET의 대사 특성과 CT의 해부학적 정밀함이 상호 보완적으로 작용했기 때문이다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 선형 계산 복잡도라는 장점을 활용하면서도, 다중 모달 의료 영상의 특성인 '모달리티별 상호 보완성'을 극대화하는 구조를 설계했다.

- **강점**: 단순히 데이터를 쌓아서 입력하는 대신, 독립적인 인코더를 통해 각 모달리티의 고유 특성을 먼저 추출하고, 이후 이중 레벨 어텐션을 통해 동적으로 가중치를 부여한 점이 성능 향상의 핵심이다. 이는 특정 모달리티에 포함된 노이즈가 다른 모달리티의 유용한 정보까지 오염시키는 것을 방지한다.
- **한계 및 논의**: 본 연구는 정량적 성능 향상에 집중했으나, 모델 내부에서 어떤 모달리티가 어떤 상황에서 더 중요하게 작용했는지에 대한 해석 가능성(Interpretability)은 충분히 다루지 않았다. 저자들 또한 이를 향후 연구 과제로 언급하며, 각 모달리티의 기여도를 수학적으로 정량화할 계획임을 밝혔다.

## 📌 TL;DR

본 논문은 3D 다중 모달 의료 영상의 종양 분할을 위해 **모달리티 전용 Mamba 인코더**와 **이중 레벨 적응형 융합 블록(Bi-level synergistic integration block)**을 제안한다. 이를 통해 Transformer의 높은 계산 비용 문제를 해결하면서도 전역적 문맥을 효과적으로 포착하였으며, $\text{BraTS2023}$ 및 $\text{Hecktor2022}$ 데이터셋에서 SOTA 성능을 달성했다. 이 연구는 효율적인 3D 의료 영상 분석을 위한 Mamba 기반의 다중 모달 융합 프레임워크의 가능성을 제시한다.
