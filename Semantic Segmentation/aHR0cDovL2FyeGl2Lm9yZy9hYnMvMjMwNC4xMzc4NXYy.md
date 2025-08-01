# Customized Segment Anything Model for Medical Image Segmentation

Kaidong Zhang and Dong Liu

## 🧩 Problem to Solve

의료 영상 분할은 컴퓨터 보조 진단 및 지능형 임상 수술에 필수적입니다. 그러나 기존 딥러닝 기반 방법들은 특정 데이터셋에 대한 광범위한 네트워크 엔지니어링을 요구하며, 모델 크기가 커서 배포 및 저장 비용이 상당합니다. DALL-E, GPT-4, SAM과 같은 대규모 모델들은 일반화 능력이 뛰어나지만, 의료 영상 분할에는 다음과 같은 한계가 있습니다:

* 대규모 모델은 주로 강도(intensity) 변화에 기반하여 분할 경계를 결정하는데, 이는 해부학적 또는 병리학적 구조 분석이 중요한 의료 영상에는 적합하지 않습니다.
* 분할 영역을 의미 있는 해부학적/병리학적 클래스와 연관시키지 못하여 의료 영상에 대한 의미론적 분할(semantic segmentation)이 불가능합니다.
이 논문은 이러한 문제들을 해결하기 위해 대규모 시각 모델인 SAM(Segment Anything Model)을 의료 영상 분할에 맞게 커스터마이징하는 방법을 모색합니다.

## ✨ Key Contributions

* SAM을 의미론적 레이블이 포함된 의료 영상 분할에 적용하고 그 가능성을 처음으로 탐구했습니다.
* 이미지 인코더의 적응과 성능, 배포, 저장 오버헤드를 고려한 일련의 미세 조정(finetuning) 전략들을 제시했습니다.
* 제안하는 SAMed는 기존의 잘 설계된 의료 영상 분할 방법들과 DSC(Dice Similarity Coefficient) 및 HD(Hausdorff Distance) 모두에서 매우 경쟁력 있는 결과를 달성했습니다.

## 📎 Related Works

* **의료 영상 분할 모델**: U-Net [29]을 시작으로 Res-UNet [35], Dense-UNet [21], U-Net++ [37], 3D-Unet [5] 등 다양한 변형이 개발되었습니다. 최근에는 Transformer 블록이 U-Net 프레임워크에 통합되어 TransUnet [4], SwinUnet [3], MiSSFormer [13], Hi-Former [10], DAE-Former [1] 등 비약적인 성능 향상을 보였습니다. 동시 연구인 MedSAM [24]도 SAM을 의료 영상에 적용했지만, SAMed는 의미론적 분할과 이미지 인코더의 특정 기능 추출에 중점을 둡니다.
* **대규모 모델**: BERT [18], GPT-4 [26], LLaMA [31]와 같은 NLP 분야와 SAM [19], SegGPT [34], STU-Net [14]과 같은 컴퓨터 비전 분야에서 놀라운 제로샷 일반화 능력을 보여주었지만, 전문 데이터 부족으로 의료 영상에 직접 적용하기는 어려움이 있습니다.
* **미세 조정 전략**: 사전 학습된 대규모 모델에 새로운 지식을 주입하기 위한 전략으로 Visual Prompt Tuning [16], LoRA(Low-Rank Adaptation) [12] 등이 제안되었습니다. SAMed 또한 LoRA를 활용하여 SAM을 의료 영상 분할에 맞게 커스터마이징합니다.

## 🛠️ Methodology

SAMed는 SAM의 아키텍처를 계승하여 의료 영상 분할에 최적화되었습니다.

* **전반적 아키텍처**:
  * 주어진 의료 영상 $x \in \mathbb{R}^{H \times W \times C}$ 에 대해, 해상도 $H \times W$인 분할 맵 $\hat{S}$를 예측하여 각 픽셀이 미리 정의된 클래스 목록 $Y = \{y_0, y_1, ..., y_k\}$ 에 속하도록 합니다 ($y_0$는 배경, $y_i$는 다른 기관 클래스).
  * SAM의 이미지 인코더는 모든 파라미터를 고정(freeze)하고, 각 트랜스포머 블록에 LoRA 기반의 학습 가능한 바이패스(bypass)를 추가합니다.
  * LoRA 레이어는 트랜스포머 특징을 저랭크 공간으로 압축한 후, 동결된 트랜스포머 블록의 출력 특징 채널에 맞춰 재투영합니다.
  * 프롬프트 인코더는 추론 시 자동 분할을 위해 어떠한 프롬프트도 필요로 하지 않으므로, SAM의 기본 임베딩을 학습 가능하게 유지하여 미세 조정합니다.
  * 마스크 디코더는 경량 트랜스포머 디코더와 분할 헤드로 구성되며, 학습 중에 미세 조정됩니다.

* **이미지 인코더의 LoRA (Low-Rank Adaptation)**:
  * 이미지 인코더의 파라미터를 모두 미세 조정하는 대신, LoRA를 사용하여 작은 부분의 파라미터만 업데이트합니다. 이는 계산 오버헤드와 배포/저장 비용을 크게 줄입니다.
  * LoRA는 투영 레이어 $W$의 업데이트 $\Delta W$를 저랭크 행렬 $BA$로 근사합니다:
        $$ \hat{W} = W + \Delta W = W + BA $$
        여기서 $A \in \mathbb{R}^{r \times C_{in}}$, $B \in \mathbb{R}^{C_{out} \times r}$ 이고 $r \ll \min\{C_{in}, C_{out}\}$ 입니다.
  * SAMed는 멀티-헤드 셀프 어텐션의 쿼리(query, $Q$) 및 값(value, $V$) 투영 레이어에 LoRA를 적용하여 어텐션 스코어에 영향을 미칩니다:
        $$ \text{Att}(Q,K,V) = \text{Softmax}(\frac{QK^T}{\sqrt{C_{out}}} + B)V $$
        $$ Q = \hat{W}_q F = W_q F + B_q A_q F $$
        $$ K = W_k F $$
        $$ V = \hat{W}_v F = W_v F + B_v A_v F $$
        여기서 $W_q, W_k, W_v$는 SAM에서 동결된 투영 레이어이며, $A_q, B_q, A_v, B_v$는 학습 가능한 LoRA 파라미터입니다.

* **프롬프트 인코더 및 마스크 디코더**:
  * **프롬프트 인코더**: 추론 시 프롬프트가 제공되지 않을 때 SAM이 사용하는 기본 임베딩을 학습 가능하게 만듭니다.
  * **마스크 디코더**: 경량 트랜스포머 레이어와 분할 헤드로 구성됩니다. SAMed는 $k$개의 의미론적 클래스에 대해 $k$개의 분할 마스크 $\hat{S}_l \in \mathbb{R}^{h \times w \times k}$를 동시에 예측합니다.
  * 최종 분할 맵은 다음처럼 생성됩니다:
        $$ \hat{S} = \text{argmax}(\text{Softmax}(\hat{S}_l, d=-1), d=-1) $$
        ($d=-1$은 마지막 차원(채널 차원)에 걸쳐 Softmax 및 argmax 연산이 수행됨을 나타냅니다.)

* **학습 전략**:
  * **손실 함수**: 크로스 엔트로피 손실(CE)과 Dice 손실(Dice)을 결합하여 사용합니다:
        $$ L = \lambda_1 \text{CE}(\hat{S}_l, D(S)) + \lambda_2 \text{Dice}(\hat{S}_l, D(S)) $$
        여기서 $D$는 다운샘플링 연산이며, $\lambda_1=0.2$, $\lambda_2=0.8$입니다.
  * **웜업(Warmup)**: 학습 초기에 학습률을 점진적으로 증가시켜 학습 과정을 안정화하고 의료 데이터에 익숙해지도록 합니다. 웜업 후에는 지수 학습률 감소를 적용합니다.
        $$ \text{lr} = \begin{cases} \frac{T}{\text{WP}} \text{lr}_{\text{I}}, & T \le \text{WP} \\ \text{lr}_{\text{I}}(1 - \frac{T-\text{WP}}{\text{MI}}), & T > \text{WP} \end{cases} $$
        ($\text{lr}_{\text{I}}$는 초기 학습률, $T$는 학습 반복 횟수, $\text{WP}$는 웜업 기간, $\text{MI}$는 최대 반복 횟수입니다.)
  * **AdamW 최적화**: AdamW [23] 옵티마이저를 사용하여 성능 개선 및 안정적인 미세 조정을 유도합니다.

## 📊 Results

* **데이터셋**: Synapse 다기관 CT 분할 데이터셋 (30개의 복부 CT 스캔).
* **정량적 비교**: SAMed는 Synapse 데이터셋에서 81.88 DSC 및 20.64 HD를 달성하여 U-Net [29], TransUnet [4], SwinUnet [3], DAE-Former [1] 등 기존 SOTA 방법들과 견줄 만한 매우 경쟁력 있는 성능을 보여주었습니다. 특히 췌장과 위 분할에서 SOTA 성능을 달성했습니다. SAMed는 SAM에 작은 파라미터 추가만으로 통합될 수 있어, 기존 의료 영상 모델들의 단점인 높은 배포 및 저장 오버헤드를 해결합니다.
* **정성적 비교**: SAMed는 TransUnet, SwinUnet, DAE-Former에 비해 더 부드럽고 정확한 분할 마스크를 생성했습니다. 이는 SAM의 강력한 특징 추출 능력과 적절한 미세 조정 전략 덕분입니다.
* **어블레이션 연구**:
  * **LoRA 미세 조정**: 마스크 디코더만 미세 조정하는 것보다 이미지 인코더와 마스크 디코더 모두를 LoRA로 미세 조정할 때 분할 정확도가 더 높았습니다 (67.95 DSC vs. 81.88 DSC). 이는 이미지 인코더의 커스터마이징이 의료 이미지 특징 추출에 중요함을 시사합니다.
  * **마스크 디코더의 LoRA 적용**: 마스크 디코더의 트랜스포머 레이어에 LoRA를 적용한 SAMed_s 버전은 SAMed보다 모델 크기가 훨씬 작지만 (6.32M vs. 18.81M), 성능은 약간 낮았습니다 (77.78 DSC vs. 81.88 DSC). 배포/저장 제약이 엄격한 경우 SAMed_s가 대안이 될 수 있습니다.
  * **LoRA 랭크 크기**: 랭크 크기 4에서 최적의 성능을 보였으며, 너무 크거나 작으면 성능이 저하되었습니다. 이는 LoRA 레이어에 필요한 최소한의 파라미터가 있지만, 너무 많으면 SAM의 본래 능력을 저해할 수 있음을 나타냅니다.
  * **LoRA 적용 투영 레이어**: 쿼리(Q)와 값(V) 투영 레이어에 LoRA를 적용하는 것이 가장 좋은 성능을 달성했으며, 모든 투영 레이어(Q, K, V, O)에 적용할 경우 성능이 크게 저하되었습니다.
  * **학습 전략 효과**: 웜업과 AdamW 옵티마이저를 적용하면 SAMed의 성능이 크게 향상되고 (56.54 DSC -> 81.88 DSC), 학습 과정이 훨씬 안정화되어 손실 값이 현저히 낮아집니다. 이러한 학습 전략은 TransUnet 및 SwinUnet과 같은 다른 의료 영상 분할 모델의 LoRA 미세 조정에도 긍정적인 영향을 미쳤습니다.

## 🧠 Insights & Discussion

* SAMed는 대규모 비전 모델인 SAM을 의료 영상 분할에 성공적으로 커스터마이징할 수 있음을 보여주었습니다. 이는 기존의 복잡한 네트워크 설계 없이도 경쟁력 있는 성능을 달성할 수 있는 새로운 패러다임을 제시합니다.
* **효율성**: SAMed는 SAM 파라미터의 극히 일부만 업데이트하므로, 배포 및 저장 오버헤드가 원본 SAM 시스템에 비해 미미합니다. 이는 실용적인 의료 환경에서 매우 중요한 장점입니다.
* **의미론적 이해**: SAMed는 단순한 경계 분할을 넘어, 각 분할 영역을 해부학적/병리학적 의미를 가진 클래스로 분류하는 의미론적 분할을 수행할 수 있습니다.
* **커스터마이징의 중요성**: 자연 이미지에 최적화된 SAM의 이미지 인코더를 의료 이미지 특성에 맞게 LoRA를 통해 미세 조정하는 것이 성능 향상에 필수적임을 입증했습니다.
* **학습 전략의 보편성**: 웜업, AdamW 옵티마이저와 같은 학습 전략은 SAMed의 안정적인 수렴과 성능 향상에 결정적인 역할을 했으며, 이는 다른 LoRA 기반 미세 조정 모델에도 적용될 수 있는 일반적인 통찰을 제공합니다.

## 📌 TL;DR

* **문제**: 대규모 일반 영상 분할 모델(SAM)은 의료 영상의 해부학적/의미론적 이해가 부족하고, 기존 의료 영상 모델은 배포 및 엔지니어링 비용이 높습니다.
* **방법**: 이 논문은 SAM의 이미지 인코더에 LoRA(Low-Rank Adaptation)를 적용하고 프롬프트 인코더 및 마스크 디코더를 미세 조정하는 `SAMed`를 제안합니다. 웜업과 AdamW 옵티마이저를 포함한 최적화된 학습 전략을 사용합니다.
* **결과**: SAMed는 Synapse 데이터셋에서 기존 SOTA 모델들과 견줄 만한 높은 성능(81.88 DSC, 20.64 HD)을 달성했으며, SAM 파라미터의 극히 일부만 업데이트하여 배포 및 저장 오버헤드를 대폭 줄였습니다. 이를 통해 SAMed는 의료 영상의 의미론적 분할을 효율적으로 수행하는 실용적인 솔루션을 제공합니다.
