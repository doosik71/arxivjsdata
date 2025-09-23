# Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang

## 🧩 Problem to Solve

최근 Mamba와 같은 효율적인 하드웨어 설계를 갖춘 상태 공간 모델(SSM)은 긴 시퀀스 모델링에서 큰 잠재력을 보여주었습니다. 이에 따라 순수 SSM 기반의 효율적이고 범용적인 비전 백본을 구축하는 것이 매력적인 방향으로 떠올랐습니다. 그러나 시각 데이터의 **위치 민감성**과 시각 이해를 위한 **전역적 맥락(global context) 요구 사항** 때문에 SSM이 시각 데이터를 표현하는 데는 어려움이 있었습니다. 이 연구는 시각 표현 학습에 있어서 셀프 어텐션(self-attention)이 필수적이지 않음을 보여주고, 기존 Transformer 모델의 **높은 계산량과 메모리 사용량** 문제를 해결하면서도 높은 성능을 유지하는 새로운 비전 백본을 제안합니다.

## ✨ Key Contributions

- **Vision Mamba (Vim) 제안**: 데이터 의존적인 전역 시각적 맥락 모델링을 위한 양방향 SSM(Bidirectional SSM)과 위치 인식 시각 이해를 위한 위치 임베딩(position embedding)을 통합한 Vision Mamba (Vim) 모델을 제안합니다.
- **효율성 및 성능 우위**: 어텐션(attention) 메커니즘 없이도 ViT와 동일한 모델링 능력을 가지며, 계산 복잡도는 이차 시간(sub-quadratic time)에서 선형 시간(linear time)으로, 메모리 복잡도는 선형(linear)으로 감소시킵니다. 특히, 1248 $\times$ 1248 해상도 이미지의 배치 추론 시 DeiT보다 2.8배 빠르고 GPU 메모리를 86.8% 절약합니다.
- **광범위한 실험 결과**: ImageNet 분류 및 다운스트림 밀집 예측(dense prediction) 작업(객체 감지, 인스턴스 분할, 시맨틱 분할)에서 기존 Transformer 모델인 DeiT보다 우수한 성능을 달성합니다.

## 📎 Related Works

- **범용 비전 백본 아키텍처**:
  - **ConvNet**: 초기 컴퓨터 비전의 표준이었으며, ResNet, Inception, DenseNet, RegNet 등의 다양한 변형이 제안되었습니다.
  - **Vision Transformer (ViT)**: 이미지를 2D 패치 시퀀스로 간주하여 순수 Transformer 아키텍처를 적용하며 큰 성공을 거두었습니다. Swin Transformer와 같은 변형은 2D 컨볼루션 사전 지식(prior)을 도입하여 효율성을 높였습니다.
  - **LongViT**: 긴 시퀀스 인코딩을 위한 효율적인 Transformer 아키텍처로, 팽창 어텐션(dilated attention)을 사용합니다.
- **긴 시퀀스 모델링을 위한 상태 공간 모델(SSM)**:
  - **S4 (Structured State-Space Sequence model)**: 장거리 의존성 모델링을 위한 CNN 및 Transformer의 대안으로 제안되었으며, 시퀀스 길이에 선형적으로 확장됩니다.
  - **Mamba**: 데이터 의존적 SSM 계층을 제안하고 하드웨어 친화적인 알고리즘을 통해 효율적인 학습 및 추론을 가능하게 하여 언어 모델링에서 Transformer의 유망한 대안으로 부상했습니다.
- **시각 애플리케이션을 위한 SSM**:
  - **1D S4**: 비디오 분류에서 장거리 시간 의존성을 처리하는 데 사용되었습니다.
  - **TranS4mer**: S4와 셀프 어텐션을 결합한 하이브리드 모델입니다.
  - **U-Mamba**: 생체 의학 이미지 분할을 위해 CNN-SSM 하이브리드 아키텍처를 제안합니다.
  - **V Mamba**: 다방향 스캐닝 및 계층적 네트워크 아키텍처를 Mamba에 통합한 동시 연구입니다. 기존 연구들이 SSM을 특정 시각 애플리케이션에 적용하거나 컨볼루션 또는 어텐션과 결합한 하이브리드 아키텍처를 구축한 것과 달리, **Vim은 순수 SSM 기반의 범용 비전 백본을 구축합니다.**

## 🛠️ Methodology

Vim은 Mamba 모델을 컴퓨터 비전에 도입하기 위해 다음 단계를 따릅니다.

1. **이미지 전처리**:

   - 입력 이미지 $I \in \mathbb{R}^{H \times W \times C}$를 겹치지 않는 $P \times P$ 크기의 2D 패치 $x_p \in \mathbb{R}^{J \times (P^2 \cdot C)}$로 분할합니다. ($J$는 패치 수)
   - 각 패치를 선형 레이어 $W \in \mathbb{R}^{(P^2 \cdot C) \times D}$를 통해 $D$ 차원 벡터로 투영합니다.
   - ViT에서 영감을 받아 학습 가능한 클래스 토큰($t_{cls}$)을 추가하고, 패치 토큰 시퀀스에 위치 임베딩 $E_{pos} \in \mathbb{R}^{(J+1) \times D}$을 더하여 최종 토큰 시퀀스 $T_0$를 생성합니다.
     $$ T*0 = [t*{cls}; t*1^p W; t_2^p W; \cdots; t_J^p W] + E*{pos} $$

2. **Vim 블록**:

   - 기존 Mamba 블록이 단방향 시퀀스에 적합했던 것과 달리, Vim 블록은 공간 인식(spatial-aware) 이해를 위해 **양방향 시퀀스 모델링**을 통합합니다.
   - 입력 토큰 시퀀스 $T_{l-1}$을 정규화합니다.
   - 정규화된 시퀀스를 각각 $x$와 $z$ 두 개의 스트림으로 선형 투영합니다.
   - **양방향 처리**: $x$ 스트림을 정방향(forward)과 역방향(backward)으로 각각 처리합니다.
     - 각 방향에서 1D 컨볼루션(Conv1d)을 적용하여 $x'_o$를 얻습니다.
     - $x'_o$를 이용하여 SSM에 필요한 파라미터 $B_o, C_o, \Delta_o$를 선형 투영합니다.
     - $\Delta_o$를 사용하여 연속 시스템 파라미터 $A_o, B_o$를 이산 시스템 파라미터 $\bar{A}_o, \bar{B}_o$로 변환합니다. (ZOH 방식 활용)
     - SSM 재귀를 통해 각 방향의 출력 $y_{forward}$ 및 $y_{backward}$를 계산합니다.
       $$ h*t = \bar{A} h*{t-1} + \bar{B} x_t $$
        $$ y_t = \bar{C} h_t $$
   - $y_{forward}$와 $y_{backward}$를 $z$ 스트림의 게이팅 유닛(SiLU(z))과 곱한 후 합산합니다.
   - 최종 결과에 잔차 연결(residual connection)을 적용하여 $T_l$을 출력합니다.

3. **Vim 인코더**:

   - 여러 개의 Vim 블록을 쌓아 Vim 인코더를 구성합니다.
   - 최종 레이어의 클래스 토큰 $T_0^L$을 정규화한 후 MLP 헤드에 전달하여 최종 예측 $\hat{p}$를 얻습니다.

4. **효율성 분석**:
   - **IO 효율성**: Mamba와 유사하게, HBM(High Bandwidth Memory)과 SRAM(Static Random-Access Memory) 간의 데이터 전송을 최적화하여 IO 수를 $O(BMEN)$에서 $O(BME+EN)$으로 줄입니다.
   - **메모리 효율성**: 역전파 시 기울기 계산에 필요한 중간 상태를 재계산하고, 활성화 함수와 컨볼루션의 중간 활성화 값도 재계산하여 GPU 메모리 사용량을 최적화합니다.
   - **계산 효율성**: 셀프 어텐션의 계산 복잡도가 시퀀스 길이 $M$에 대해 이차($O(M^2 D)$)인 반면, Vim의 SSM은 시퀀스 길이에 대해 선형($O(M D N)$) 복잡도를 가집니다($N$은 SSM 차원으로 고정). 이는 긴 시퀀스에 대한 확장성을 제공합니다.

## 📊 Results

- **ImageNet 분류**:
  - Vim은 DeiT의 다양한 스케일(Tiny, Small, Base)에서 동등하거나 더 적은 파라미터 수로 DeiT보다 높은 Top-1 정확도를 달성했습니다 (예: Vim-Tiny가 DeiT-Tiny보다 3.9%p 높음).
  - SSM 기반 S4ND-ViT-B와 비교하여 Vim은 3배 적은 파라미터로 유사한 정확도를 달성했습니다.
  - 긴 시퀀스 미세 조정(fine-tuning) 후 Vim 모델들은 더 높은 정확도를 보여, 긴 시퀀스 모델링 능력과 강력한 시각 표현 추출 능력을 입증했습니다.
- **효율성 (속도 및 메모리)**:
  - DeiT와 Vim-Ti의 FPS 및 GPU 메모리 비교에서, 이미지 해상도가 높아질수록 Vim의 효율성이 크게 향상됩니다.
  - $1248 \times 1248$ 해상도에서 Vim은 DeiT보다 2.8배 빠르고 GPU 메모리를 86.8% 절약했습니다.
  - 다운스트림 작업(분할, 감지)에서도 FPN(Feature Pyramid Network)을 부착했음에도 불구하고 유사한 효율성 우위를 유지했습니다.
- **시맨틱 분할 (ADE20K)**:
  - UperNet 프레임워크를 사용했을 때, Vim은 DeiT보다 모든 스케일에서 높은 mIoU를 달성했습니다 (예: Vim-Ti가 DeiT-Ti보다 1.8 mIoU 높음).
  - ResNet-101 백본과 비교하여 Vim-S는 약 2배 적은 파라미터로 동일한 분할 성능을 달성했습니다.
- **객체 감지 및 인스턴스 분할 (COCO)**:
  - Cascade Mask R-CNN 프레임워크를 사용했을 때, Vim-Ti는 DeiT-Ti보다 1.3 box AP와 1.1 mask AP가 높았습니다.
  - 특히 중간 및 대형 객체에서 더 나은 성능을 보여 장거리 맥락 학습 능력을 입증했습니다. DeiT가 2D 윈도우 어텐션과 같은 2D 사전 지식을 도입해야 하는 반면, Vim은 순수 시퀀스 모델링 방식으로 고해상도 이미지를 처리할 수 있었습니다.
- **어블레이션 연구**:
  - **양방향 SSM**: "Bidirectional SSM + Conv1d" 전략이 ImageNet 분류(73.9% Top-1 Acc) 및 ADE20K 시맨틱 분할(35.9 mIoU)에서 가장 우수한 성능을 보여, 밀집 예측 작업에서 양방향성의 중요성을 강조했습니다.
  - **분류 설계**: "Middle class token" 전략이 76.1%의 최고 Top-1 정확도를 달성하며, SSM의 반복적인 특성과 ImageNet의 중앙 객체 사전 지식을 잘 활용함을 보여주었습니다.

## 🧠 Insights & Discussion

- **Transformer의 대안**: Vim은 순수 SSM 기반으로 시각 표현 학습에 있어 셀프 어텐션이 필수적이지 않음을 성공적으로 입증했습니다. 이는 고해상도 이미지 처리 시 Transformer의 계산 및 메모리 제약을 극복할 수 있는 새로운 가능성을 제시합니다.
- **효율성과 확장성**: Mamba의 하드웨어 인식 설계 덕분에 Vim은 고해상도 이미지 처리 시 ViT보다 추론 속도와 메모리 사용량에서 현저히 우수합니다. 이러한 선형적인 확장성 덕분에 기가픽셀 수준의 이미지(항공 이미지, 의료 이미지 등)나 긴 비디오와 같은 고해상도 시각 데이터를 엔드투엔드(end-to-end) 방식으로 학습하는 데 매우 적합합니다.
- **범용 백본의 잠재력**: 데이터 의존적인 전역 시각적 맥락을 학습하는 양방향 SSM 모델링 방식을 통해 Vim은 Transformer와 유사한 모델링 능력을 가지며, 차세대 비전 백본으로서 큰 잠재력을 가집니다.
- **향후 연구 방향**: 위치 임베딩을 포함한 양방향 SSM 모델링은 마스크 이미지 모델링(MIM)과 같은 비지도 학습 작업이나 CLIP 스타일의 멀티모달 사전 학습에 적합하며, 사전 학습된 Vim 가중치를 사용하여 고해상도 의료 이미지, 원격 감지 이미지, 긴 비디오 분석과 같은 다운스트림 작업에 활용할 수 있습니다.

## 📌 TL;DR

Vision Mamba (Vim)는 시각 데이터의 위치 민감성과 전역적 맥락 요구 사항으로 인한 기존 SSM의 한계를 극복하고, Transformer의 고비용 문제를 해결하기 위해 **양방향 상태 공간 모델(SSM)**을 기반으로 하는 효율적이고 범용적인 비전 백본을 제안합니다. Vim은 이미지를 패치 시퀀스로 변환하고 양방향 SSM 블록을 통해 데이터 의존적 전역 시각적 맥락과 위치 인식을 학습합니다. 결과적으로 Vim은 ImageNet 분류 및 다양한 다운스트림 밀집 예측 작업에서 DeiT와 같은 Transformer 모델보다 **우수한 성능**을 달성하며, 특히 고해상도 이미지 처리에서 DeiT 대비 **훨씬 빠른 추론 속도와 낮은 메모리 사용량**을 보여줍니다. 이는 Vim이 차세대 비전 백본으로서 높은 잠재력을 가지고 있음을 시사합니다.
