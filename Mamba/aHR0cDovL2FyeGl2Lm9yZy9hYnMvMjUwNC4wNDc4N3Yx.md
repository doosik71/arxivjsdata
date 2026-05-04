# Dynamic Vision Mamba

Mengxuan Wu, Zekai Li, Zhiyuan Liang, Moyang Li, Xuanlei Zhao, Samir Khaki, Zheng Zhu, Xiaojiang Peng, Konstantinos N. Plataniotis, Kai Wang, Wangbo Zhao, Yang You (2025)

## 🧩 Problem to Solve

본 논문은 Mamba 기반의 비전 모델에서 발생하는 **공간적 중복성(Spatial Redundancy)** 문제를 해결하고자 한다. 이 중복성은 크게 두 가지 수준에서 나타난다.

첫째는 **토큰 수준의 중복성(Token Redundancy)**이다. 이미지 내의 많은 픽셀이나 토큰이 모델 성능에 기여하는 바가 매우 적음에도 불구하고, 모든 토큰을 처리함으로써 계산 비용이 증가하고 추론 속도가 저하된다. 특히 기존의 Vision Transformer(ViT)에서 사용되던 토큰 프루닝(Token Pruning) 방식을 Mamba에 그대로 적용할 경우, Mamba의 순환 구조(Recurrent-like structure) 특성상 학습 시의 마스킹과 추론 시의 토큰 제거 사이에 상태 전이 횟수(Evolution transformations, $\bar{A}$)의 차이가 발생하여 **학습-추론 불일치(Training-Inference Inconsistency)** 문제가 발생한다.

둘째는 **블록 수준의 중복성(Block Redundancy)**이다. 많은 Vision Mamba 모델(예: Vim)은 공간적 인식을 높이기 위해 레이어마다 순방향(Forward) 및 역방향(Backward) SSM 블록을 모두 사용한다. 하지만 실험적으로 SSM 블록의 수가 추론 처리량(Throughput)에 매우 큰 영향을 미친다는 점이 확인되었으며, 모든 이미지에 대해 모든 블록을 사용하는 것은 비효율적이다.

따라서 본 연구의 목표는 모델의 성능 저하를 최소화하면서 FLOPs를 줄이고 추론 속도를 높이기 위해, 토큰과 블록 수준의 중복성을 동적으로 제거하는 **Dynamic Vision Mamba (DyVM)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 구조에 최적화된 동적 프루닝 전략을 통해 계산 효율성을 극대화한 것이다.

1.  **토큰 재배치 전략(Token Rearrangement Strategy):** 학습 과정에서 프루닝될 토큰들을 별도로 모아 재배치함으로써, 보존된 토큰들이 추론 단계와 동일한 상태 전이 경로를 갖도록 설계하였다. 이를 통해 추가적인 계산 비용 없이 학습-추론 간의 일관성을 확보하였다.
2.  **동적 블록 선택(Dynamic Block Selection):** 각 입력 이미지의 특성에 따라 필요한 SSM 블록(순방향, 역방향 또는 둘 다/없음)을 동적으로 선택하는 메커니즘을 도입하여 추론 처리량을 획기적으로 개선하였다.
3.  **통합 프루닝 프레임워크:** 토큰과 블록 수준의 중복성을 동시에 해결하는 DyVM을 통해, Vim-S 모델 기준 성능 하락을 1.7%로 억제하면서 FLOPs를 35.2% 감소시키는 성과를 거두었다.

## 📎 Related Works

**Mamba for Vision:** 최근 State Space Models (SSMs) 기반의 Mamba가 긴 시퀀스를 효율적으로 처리할 수 있다는 점이 입증되며, Vim, VMamba, PlainMamba 등 다양한 비전 모델이 제안되었다. 이들은 주로 2D 이미지를 1D 시퀀스로 변환하여 SSM을 통해 토큰 간 상호작용을 모델링한다.

**Token Pruning:** ViT에서는 EViT, DynamicViT, ToMe 등 중요도가 낮은 토큰을 제거하여 연산량을 줄이는 기법들이 연구되었다. 하지만 Mamba는 토큰의 순서와 상태 전이가 중요한 순차적 구조를 가지므로, ViT의 단순 마스킹 방식을 적용하면 성능이 급격히 저하된다. 최근 HiddenAlign(HA)이 Mamba의 토큰 프루닝을 시도했으나, 추론 시에 프루닝된 토큰의 상태 전이($\bar{A}$)를 유지해야 하므로 추가적인 계산 오버헤드가 발생하는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인
DyVM은 특정 레이어 사이에 **Predictor**를 배치하여 토큰을 점진적으로 프루닝하고, 매 레이어마다 **Block Selector**를 통해 사용할 SSM 블록을 결정하는 구조를 가진다.

### 토큰 프루닝 및 재배치 (Token Pruning & Rearrangement)
토큰 프루닝은 $S$번의 단계(Stage)에 걸쳐 점진적으로 수행된다.

1.  **예측 및 마스킹:** Predictor $P$가 각 토큰의 보존 확률을 계산하고, Gumbel-Softmax를 통해 이진 마스크 $M^s \in \{0, 1\}$를 생성한다.
    $$\Pi = \text{Softmax}(P(H, M^{s-1}))$$
    $$\hat{M} = \text{Gumbel-Softmax}(\Pi)$$
    $$M^s = \hat{M} \odot M^{s-1}$$

2.  **재배치(Rearrangement) 과정:** 학습-추론 불일치를 해결하기 위해 다음과 같이 토큰을 재배치한다.
    *   보존된 토큰들을 상대적 순서를 유지한 채 하나의 연속된 블록 $X_{retained}$로 모은다.
    *   이때 Class Token $c$를 Vim 모델의 설정에 맞게 중간에 삽입한다.
    *   프루닝된 토큰들을 별도의 블록 $X_{pruned}$로 모아 보존된 블록 뒤에 연결한다.
    $$X_{rearranged} = \text{Concat}(X_{retained}, X_{pruned})$$
    이 방식을 통해 보존된 토큰들은 추론 때와 동일하게 $i$번째 위치에서 $i$번의 상태 전이를 겪게 되어 일관성이 유지된다.

### 동적 블록 선택 (Dynamic Block Selection)
각 레이어 $l$에서 Class Token $C^l$을 입력으로 받아 어떤 SSM 블록을 사용할지 결정하는 블록 선택기 $G$를 사용한다.

1.  **블록 마스크 생성:** Gumbel-Sigmoid를 통해 각 블록의 활성화 여부를 결정하는 이진 마스크 $Q^l \in \{0, 1\}^2$를 생성한다.
    $$Q^l = \text{Gumbel-Softmax}(G(C^l))$$
2.  **블록 활성화:** 생성된 마스크를 각 블록의 출력에 곱하여 불필요한 블록의 출력을 제거한다.
    $$O_{l,f} = \text{Forward-Block}(H^l) \cdot Q^l_{:,0}$$
    $$O_{l,b} = \text{Backward-Block}(H^l) \cdot Q^l_{:,1}$$

### 학습 목표 및 손실 함수 (Training Objectives)
모델은 다음 다섯 가지 손실 함수의 가중 합으로 학습된다.
$$L_{joint} = \lambda_{cls}L_{cls} + \lambda_{token}L_{token} + \lambda_{block}L_{block} + \lambda_{dis\_out}L_{dis\_out} + \lambda_{dis\_token}L_{dis\_token}$$

*   **$L_{cls}$:** 표준 Cross-Entropy 분류 손실.
*   **$L_{token}, L_{block}$:** 목표 프루닝 비율 $\rho$와 실제 프루닝 비율 간의 차이를 줄이기 위한 MSE 손실.
*   **$L_{dis\_out}$:** Teacher 모델(원본 Backbone)의 출력과 Student 모델의 출력 간의 KL-Divergence 손실.
*   **$L_{dis\_token}$:** Teacher와 Student의 보존된 토큰 표현 간의 MSE 손실.

## 📊 Results

### 실험 설정
*   **모델:** Vim (T, S, B), VideoMamba, MambaReg.
*   **데이터셋:** ImageNet-1K (분류), Kinetics-400 (비디오 이해), ADE20K (시맨틱 세그멘테이션).
*   **지표:** Top-1 Accuracy, FLOPs, Throughput (img/sec), mIoU.

### 주요 결과
1.  **이미지 분류 (ImageNet-1K):**
    *   Vim-S 모델에서 FLOPs를 **35.2% 감소**시키면서 정확도 손실은 단 **1.7%**에 그쳤다.
    *   HiddenAlign(HA)과 비교했을 때, DyVM은 유사하거나 더 높은 정확도를 유지하면서 더 많은 FLOPs를 절감하였다.
2.  **처리량(Throughput) 개선:**
    *   다양한 GPU(V100, A6000, A100)에서 처리량이 크게 향상되었다. 특히 모델 크기가 클수록(Vim-B) 개선 폭이 컸으며, A100 기준 Vim-B의 처리량이 약 45.8% 증가하였다.
3.  **일반화 성능:**
    *   VideoMamba와 MambaReg에도 적용 가능함을 보였으며, 특히 비디오 이해 작업(Kinetics-400)에서도 FLOPs를 36-38% 줄이며 유효한 성능을 유지하였다.
    *   시맨틱 세그멘테이션(ADE20K) 작업에서도 mIoU 하락을 최소화하며 연산량을 줄였다.

## 🧠 Insights & Discussion

**강점 및 분석:**
*   **학습-추론 일관성의 효율적 해결:** 기존 HA 방식은 추론 시 불필요한 연산을 유지해야 했으나, DyVM의 재배치 전략은 학습 시에만 데이터를 조정함으로써 추론 오버헤드를 완전히 제거하였다.
*   **모델 크기와 중복성의 관계:** 실험 결과, 모델의 크기가 클수록(Vim-T $\rightarrow$ Vim-S $\rightarrow$ Vim-B) 프루닝에 의한 성능 저하가 적게 나타났다. 이는 대형 모델이 더 많은 공간적 중복성을 가지고 있어 공격적인 프루닝을 더 잘 견딜 수 있음을 시사한다.
*   **학습 가능한 Predictor의 우수성:** 무작위 선택이나 고정 위치 프루닝보다 학습 가능한 Predictor를 사용했을 때 정확도가 훨씬 높았다. 이는 모델이 데이터에 따라 어떤 토큰과 블록이 중요한지 정확히 학습했음을 의미한다.

**한계 및 논의:**
*   **프루닝 단계의 영향:** 토큰을 한 번에 제거하는 것보다 3단계에 걸쳐 점진적으로 제거하는 것이 성능 유지에 유리함을 확인하였다. 이는 급격한 정보 손실을 방지하는 완충 작용이 필요함을 보여준다.
*   **Dense Prediction의 제약:** 시맨틱 세그멘테이션과 같은 작업에서는 픽셀 단위의 맵이 필요하므로, 토큰을 완전히 삭제하는 대신 업데이트를 중단하고 나중에 원래 위치로 복원하는 추가적인 처리가 필요했다.

## 📌 TL;DR

본 논문은 Mamba 기반 비전 모델의 연산 효율성을 높이기 위해 **토큰과 블록 수준의 중복성을 동적으로 제거하는 DyVM**을 제안한다. 특히 Mamba 특유의 순차 구조로 인해 발생하는 학습-추론 불일치 문제를 **토큰 재배치 전략**으로 해결하였으며, 이미지별 최적의 SSM 블록을 선택하는 **동적 블록 선택**을 통해 추론 속도를 획기적으로 개선하였다. 결과적으로 Vim-S 기준 FLOPs를 35.2% 줄이면서도 성능 저하를 최소화하였으며, 다양한 Mamba 아키텍처와 비전 작업으로의 높은 일반화 가능성을 입증하였다.