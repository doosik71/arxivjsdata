# InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions

Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, Xiaogang Wang, Yu Qiao

---

## 🧩 Problem to Solve

최근 대규모 비전 트랜스포머(ViT)가 컴퓨터 비전 분야에서 큰 성공을 거두었지만, 컨볼루션 신경망(CNN) 기반의 대규모 모델은 여전히 초기 단계에 머물러 있었습니다. 기존 CNN은 제한된 유효 수용 영역(effective receptive field)과 고정된 공간 집계 방식으로 인해 ViT와 달리 대규모 파라미터와 데이터를 효과적으로 활용하여 강력하고 견고한 특징을 학습하는 데 한계가 있었습니다. 이 연구는 CNN 기반 모델도 ViT에 필적하거나 그 이상의 성능을 달성할 수 있음을 보여주고자 합니다.

## ✨ Key Contributions

* **새로운 대규모 CNN 기반 기반 모델 제시**: InternImage는 10억 개 이상의 파라미터와 4억 개 이상의 훈련 이미지를 효과적으로 확장하여 최첨단 ViT와 필적하거나 그 이상의 성능을 달성한 최초의 CNN 기반 모델입니다. 이는 컨볼루션 모델 또한 대규모 모델 연구의 유효한 방향임을 입증합니다.
* **개선된 변형 가능한 컨볼루션 연산자 개발**: 3x3 DCN(Deformable Convolution) 연산자를 개선하여 CNN에 장거리 의존성 및 적응형 공간 집계를 성공적으로 도입하고, 이를 중심으로 한 기본 블록, 스태킹 규칙, 스케일링 전략을 탐색했습니다. 이 설계는 대규모 파라미터 및 데이터를 통해 성능 향상을 얻을 수 있도록 합니다.
* **광범위한 벤치마크에서의 우수성 입증**: ImageNet, COCO, ADE20K 등 다양한 비전 벤치마크에서 InternImage의 효과를 입증했습니다.
  * InternImage-H는 COCO test-dev에서 65.4 mAP, ADE20K에서 62.9 mIoU를 달성하며 기존 CNN 및 ViT의 기록을 경신했습니다.

## 📎 Related Works

* **기존 CNN 모델**: AlexNet, VGG, GoogleNet, ResNet, EfficientNet 등 다양한 아키텍처 설계와 깊이별 컨볼루션(depth-wise convolution), 변형 가능한 컨볼루션(deformable convolution) 등의 연산자 개선 연구. 최근에는 대규모 커널(large kernels)이나 동적 가중치(dynamic weights)를 사용하여 장거리 의존성을 도입하려는 ConvNeXt, HorNet, SLaK, RepLKNet 등의 시도가 있었습니다.
* **비전 트랜스포머(ViT)**: ViT는 전역 수용 영역과 동적 공간 집계를 통해 비전 작업에서 큰 성공을 거두었습니다. 하지만 높은 계산 및 메모리 복잡성으로 인해 PVT, Linformer, DAT, HaloNet, Swin Transformer 등 전역 어텐션의 한계를 극복하기 위한 다양한 노력이 있었습니다.
* **대규모 모델 스케일링**: NLP 분야의 성공에서 영감을 받아 Zhai et al.은 20억 파라미터 ViT를, Liu et al.은 30억 파라미터 Swin Transformer v2를 제시했습니다. BEiT-3와 같은 멀티모달 사전 학습 모델과 ViT-CNN 하이브리드 모델도 연구되었습니다.

## 🛠️ Methodology

본 연구는 대규모 CNN 기반 기반 모델 설계를 위해 유연한 컨볼루션 변형인 **Deformable Convolution v2 (DCNv2)**를 기반으로 시작하여, 이를 대규모 기반 모델 요구사항에 맞게 개선한 **DCNv3**를 핵심 연산자로 사용합니다.

1. **DCNv3 확장**: DCNv2의 한계를 극복하고 ViT의 장점인 장거리 의존성 및 적응형 공간 집계를 도입하기 위해 다음과 같은 세 가지 주요 수정을 적용합니다.
    * **컨볼루션 뉴런 간 가중치 공유**: 기존 DCNv2는 각 샘플링 포인트에 독립적인 선형 투영 가중치를 가지지만, DCNv3는 separable convolution 아이디어에서 영감을 받아 가중치 $w_k$를 depth-wise 부분과 point-wise 부분으로 분리하고, point-wise 부분은 샘플링 포인트 간에 공유합니다. 이는 파라미터 및 메모리 효율성을 크게 향상시킵니다.
    * **멀티-그룹 메커니즘 도입**: 트랜스포머의 멀티-헤드 어텐션(MHSA)과 유사하게 공간 집계 프로세스를 $G$개의 그룹으로 분할합니다. 각 그룹은 개별적인 샘플링 오프셋 $\Delta p_{gk}$와 변조 스케일 $m_{gk}$를 가져 다른 공간 집계 패턴을 학습할 수 있게 하여 더 강력한 특징을 얻습니다.
    * **샘플링 포인트에 대한 변조 스칼라 정규화**: 기존 시그모이드 함수를 통한 요소별 정규화 대신, 샘플링 포인트 $K$를 따라 소프트맥스(softmax) 정규화를 적용하여 변조 스칼라의 합을 1로 고정함으로써 대규모 모델 훈련 시 안정성을 확보합니다.
    * 수정된 DCNv3는 다음 방정식으로 표현됩니다:
        $$y(p_0) = \sum_{g=1}^{G} \sum_{k=1}^{K} w_g m_{gk} x_g(p_0 + p_k + \Delta p_{gk})$$
        여기서 $w_g$는 그룹별 투영 가중치, $m_{gk}$는 소프트맥스 정규화된 변조 스칼라, $\Delta p_{gk}$는 입력에 따라 학습되는 오프셋입니다.

2. **InternImage 모델 아키텍처**:
    * **기본 블록**: 기존 CNN의 병목 블록(bottleneck block)과 달리 ViT와 유사하게 Layer Normalization (LN), Feed-Forward Network (FFN), GELU 활성화 함수를 포함합니다. 핵심 연산자는 DCNv3이며, 샘플링 오프셋과 변조 스칼라는 separable convolution으로 예측됩니다.
    * **Stem & 다운샘플링 레이어**: 계층적 특징 맵을 얻기 위해 컨볼루션 기반의 stem 레이어(초기 해상도를 4배 감소)와 다운샘플링 레이어(단계 사이에서 해상도를 2배 감소)를 사용합니다.
    * **스태킹 규칙**: 4단계 모델의 복잡한 하이퍼파라미터 공간을 줄이기 위해 4가지 규칙을 정의합니다. (1) $C_i$는 $C_1$에 의해 결정, (2) $G_i$는 $C_i$에 따라 결정, (3) $L_1=L_2=L_4$, (4) $L_1 \le L_3$. 이를 통해 4개의 하이퍼파라미터($C_1, C', L_1, L_3$)만으로 모델 변형을 정의합니다.
    * **스케일링 규칙**: 최적의 기본 모델을 바탕으로 모델의 깊이($D$)와 너비($C_1$)를 $\alpha, \beta$ 및 복합 계수 $\phi$를 사용하여 확장합니다 ($D' = \alpha^\phi D$, $C'_1 = \beta^\phi C_1$). 실험적으로 $\alpha=1.09$, $\beta=1.36$이 최적의 스케일링 설정임을 발견했습니다.

## 📊 Results

InternImage는 다양한 비전 태스크에서 최첨단 CNN 및 ViT 모델에 필적하거나 이를 능가하는 성능을 보였습니다.

* **ImageNet 이미지 분류**:
  * InternImage-T는 ConvNeXt-T보다 1.4점 높은 83.5% top-1 정확도를 달성했습니다.
  * InternImage-B는 하이브리드 ViT인 CoAtNet-2를 0.8점 앞섰습니다.
  * InternImage-H는 10억 개의 파라미터와 4억 2,700만 개의 대규모 데이터셋으로 사전 학습되어 89.6%의 top-1 정확도를 달성하며, 기존 CNN 모델을 능가하고 최첨단 ViT 모델에 근접했습니다.

* **COCO 객체 탐지 및 인스턴스 분할**:
  * Mask R-CNN을 사용한 1x 스케줄에서 InternImage-T는 Swin-T보다 4.5점 높은 47.2 APb를 기록했습니다.
  * 가장 큰 모델인 InternImage-H는 DINO 검출기와 결합하여 COCO test-dev에서 65.4% box mAP를 달성했으며, 이는 FD-SwinV2-G보다 1.2점 높은 결과이며, 파라미터 수는 27% 적습니다.

* **ADE20K 의미론적 분할**:
  * UperNet을 사용했을 때, InternImage-B는 ConvNeXt-B (50.8 vs. 49.1 mIoU) 및 RepLKNet-31B (50.8 vs. 49.9 mIoU)를 능가하는 50.8 mIoU를 달성했습니다.
  * InternImage-H는 Mask2Former와 다중 스케일 테스트를 사용하여 62.9 mIoU를 달성, BEiT-3를 넘어서는 최고 기록을 세웠습니다.

* **어블레이션 연구**:
  * DCNv3에서 컨볼루션 뉴런 간 가중치 공유는 파라미터 및 메모리 사용량을 크게 줄이면서도 성능 저하가 없음을 입증했습니다.
  * 멀티-그룹 공간 집계는 ImageNet에서 1.2점, COCO에서 3.4점의 AP 향상을 가져오며 강력한 특징 학습에 기여했습니다.
  * DCNv3의 3x3 커널이 대규모 유효 수용 영역을 학습하기에 충분하며, 5x5 또는 7x7과 같은 더 큰 커널은 최적화의 어려움과 성능 저하를 보였습니다.

* **강건성 및 데이터 효율성**:
  * InternImage는 ConvNeXt, PVTv2, Swin 등 다른 모델에 비해 이동(translation), 회전(rotation), 스케일링(scaling) 변환에 대해 더 뛰어난 강건성을 보였습니다.
  * 데이터 스케일에 대한 강건성 실험에서, InternImage는 1% 및 10%의 적은 데이터에서도 ResNet-50과 ConvNeXt-T보다 우수한 성능을 보였고, 전체 데이터에서는 최상위 성능을 유지했습니다.

## 🧠 Insights & Discussion

* **CNN의 잠재력 재확인**: InternImage는 Deformable Convolution을 핵심 연산자로 사용하여 CNN 기반 모델도 ViT처럼 대규모 파라미터와 데이터로부터 이득을 얻고, 비전 기반 모델로서의 역할을 성공적으로 수행할 수 있음을 입증했습니다. 이는 대규모 비전 모델 연구에서 CNN이 여전히 중요한 선택지임을 시사합니다.
* **변형 가능한 컨볼루션의 중요성**: DCNv3는 전통적인 CNN의 엄격한 귀납적 편향을 완화하고, ViT의 적응형 공간 집계 및 장거리 의존성 모델링 능력을 모방함으로써 대규모 데이터에서 더 강력하고 견고한 패턴을 학습할 수 있게 했습니다.
* **확장성 및 효율성**: 제안된 DCNv3의 가중치 공유 및 소프트맥스 정규화는 모델의 효율적인 스케일링을 가능하게 했습니다.
* **한계**: DCN 기반 연산자는 고속 처리가 요구되는 다운스트림 태스크(예: 실시간 애플리케이션)에서 여전히 지연(latency) 문제가 있습니다. 대규모 CNN 연구는 아직 초기 단계이며, InternImage가 향후 연구의 좋은 출발점이 되기를 기대합니다.

## 📌 TL;DR

InternImage는 DCNv3를 핵심 연산자로 활용하여 장거리 의존성과 적응형 공간 집계 능력을 갖춘 새로운 대규모 CNN 기반 기반 모델입니다. 이는 전통적인 CNN의 귀납적 편향을 완화하여 대규모 파라미터와 데이터를 효과적으로 학습할 수 있게 합니다. ImageNet, COCO, ADE20K 벤치마크에서 최첨단 ViT 모델과 필적하거나 이를 능가하는 성능을 달성하여, CNN이 대규모 비전 기반 모델 연구의 유효한 방향임을 입증했습니다.
