# Context Encoding for Semantic Segmentation

Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal

## 🧩 Problem to Solve

최근 완전 합성곱 네트워크(FCN) 기반의 픽셀 단위 레이블링 방법들은 공간 해상도 개선에 상당한 진전을 보였지만, 확장(Dilated/Atrous) 합성곱 사용 등으로 인해 픽셀들이 전역 장면 컨텍스트로부터 고립되어 오분류를 초래하는 문제가 발생합니다. 이 논문은 전역 컨텍스트 정보의 부족이 야기하는 semantic segmentation의 한계를 극복하고자 합니다. 즉, 수용 영역(receptive field)을 단순히 늘리는 것이 아니라, 장면 컨텍스트와 객체 카테고리 확률 간의 강한 상관관계를 네트워크가 명시적으로 활용하도록 하여, 모호한 픽셀이나 작은 객체의 분류 정확도를 높이는 것이 목표입니다.

## ✨ Key Contributions

- **Context Encoding Module (CEM) 도입:** 장면의 의미론적 컨텍스트를 효과적으로 캡처하고, 이를 바탕으로 클래스 종속 피처맵(featuremap)을 선택적으로 강조하는 새로운 모듈을 제안합니다. 이 모듈은 FCN에 비해 미미한 추가 계산 비용으로 semantic segmentation 성능을 크게 향상시킵니다.
- **Semantic Encoding Loss (SE-loss) 제안:** 픽셀 단위 분할 손실 외에, 네트워크가 전역 의미론적 컨텍스트를 학습하도록 강제하는 정규화 손실을 도입합니다. 이 손실은 크고 작은 객체에 동일한 기여를 하며, 특히 작은 객체의 분할 성능 개선에 기여합니다.
- **Context Encoding Network (EncNet) 설계 및 구현:** 사전 학습된 ResNet 기반에 Context Encoding Module을 통합한 새로운 semantic segmentation 프레임워크인 EncNet을 제안합니다.
- **최첨단 성능 달성:** PASCAL-Context에서 51.7% mIoU, PASCAL VOC 2012에서 85.9% mIoU를 달성하며 새로운 최고 성능을 기록했습니다. ADE20K 테스트 세트에서는 0.5567의 점수로 COCO-Place Challenge 2017의 우승 기록을 넘어섰습니다.
- **얕은 네트워크의 특징 표현력 향상:** CIFAR-10 이미지 분류 태스크에서 Context Encoding Module이 14개 계층의 비교적 얕은 네트워크의 오류율을 3.45%까지 낮추며, 이는 10배 이상 깊은 최첨단 네트워크들과 견줄 만한 성능입니다.
- **시스템 코드 공개:** 동기화된 멀티 GPU 배치 정규화(Synchronized Multi-GPU Batch Normalization) 및 메모리 효율적인 Encoding Layer 구현을 포함한 전체 시스템의 소스 코드를 공개했습니다.

## 📎 Related Works

- **Semantic Segmentation:**
  - **FCN 기반 접근 방식:** Long et al. [37]의 FCN은 픽셀 단위 예측을 위한 시대를 열었습니다.
  - **공간 해상도 보존:** Dilated/Atrous Convolution [4, 54]은 수용 영역을 유지하면서 밀집 예측을 생성하는 데 사용됩니다.
  - **컨텍스트 정보 확대:** PSPNet [59]의 Spatial Pyramid Pooling과 Deeplab [5]의 Atrous Spatial Pyramid Pooling은 다중 해상도 피라미드 기반 표현을 통해 수용 영역을 확장합니다.
  - **경계 정제:** Dense CRF [5, 7] 및 CRF-RNN [60]은 FCN 출력의 경계를 정제하는 데 사용됩니다.
- **Featuremap Attention 및 Scaling:**
  - Spatial Transformer Network [24], Batch Normalization [23]은 데이터의 평균과 분산을 정규화하여 학습을 가속화합니다.
  - 스타일 전이(style transfer) 연구 [11, 22, 57]는 피처맵 통계를 조작합니다.
  - SE-Net [20]은 채널 간 정보를 탐색하여 채널별 어텐션(attention)을 학습합니다.
- **고전적인 컨텍스트 인코딩:**
  - Bag-of-Words (BoW) [8, 13, 26, 46], VLAD [25], Fisher Vector [44]와 같은 고전적인 인코더는 전역 특징 통계를 캡처하여 컨텍스트 정보를 인코딩합니다.
  - Zhang et al. [58]의 Encoding Layer는 전체 딕셔너리 학습 및 잔차 인코딩 파이프라인을 단일 CNN 계층으로 통합하여 순서에 무관한 표현을 캡처합니다. 이 논문은 이를 semantic context 이해를 위한 전역 특징 통계 캡처에 확장합니다.

## 🛠️ Methodology

- **Context Encoding Module (CEM):**
  - **Encoding Layer [58]:** 입력 피처맵 $X \in \mathbb{R}^{C \times H \times W}$를 $N = H \times W$개의 $C$-차원 특징 벡터 $X=\{x_1, \dots, x_N\}$로 간주합니다. 이 계층은 $K$개의 코드워드 딕셔너리 $D=\{d_1, \dots, d_K\}$와 각 시각 중심에 대한 스무딩 팩터 $S=\{s_1, \dots, s_K\}$를 학습합니다.
  - **잔차 인코더 계산:** 각 특징 $x_i$와 코드워드 $d_k$ 사이의 잔차 $r_{ik} = x_i - d_k$를 계산하고, 소프트 할당 가중치 $e_{ik} = \frac{\exp(-s_k \|r_{ik}\|^2)}{\sum_{j=1}^K \exp(-s_j \|r_{ij}\|^2)}$를 사용하여 잔차를 집계하여 잔차 인코더 $e_k = \sum_{i=1}^N e_{ik}$를 생성합니다. 최종 인코더 $e = \sum_{k=1}^K \phi(e_k)$는 배치 정규화(Batch Normalization)와 ReLU 활성화 함수 $\phi$를 적용하여 차원을 줄입니다.
  - **Featuremap Attention:** Encoding Layer의 출력(인코딩된 의미론 $e$) 위에 완전 연결 계층과 시그모이드 활성화 함수를 사용하여 피처맵 스케일링 팩터 $\gamma = \delta(We)$를 예측합니다. 이 $\gamma$는 입력 피처맵 $X$와 채널별 곱셈($\otimes$)을 통해 $Y = X \otimes \gamma$로 모듈 출력을 형성하며, 클래스 종속 피처맵을 선택적으로 강조하거나 약화시킵니다.
  - **Semantic Encoding Loss (SE-loss):** Encoding Layer 위에 추가적인 완전 연결 계층과 시그모이드 활성화 함수를 구축하여 장면 내 객체 카테고리의 존재 여부를 예측하고 이진 교차 엔트로피 손실로 학습합니다. 이는 픽셀 단위 손실과 달리 크고 작은 객체에 동일한 기여를 하도록 설계되었습니다.
- **Context Encoding Network (EncNet) 구조:**
  - 사전 학습된 ResNet [17]을 백본(backbone)으로 사용하며, ResNet의 3단계와 4단계에 Dilated Network 전략 [4, 54]을 적용하여 출력 해상도를 1/8로 유지합니다.
  - 제안된 Context Encoding Module은 최종 예측 직전의 합성곱 계층 위에 추가됩니다.
  - SE-loss는 인코딩된 의미론을 입력으로 받아 객체 클래스의 존재를 예측하는 별도의 브랜치로 구성되어 학습을 정규화합니다. Stage 3에도 추가 Context Encoding Module을 사용하여 추가 정규화를 수행합니다.
  - EncNet은 기존 FCN 파이프라인에 미분 가능하게 삽입되며, 원래 네트워크에 비해 미미한 추가 계산 비용만 발생합니다.
- **훈련 전략:**
  - PyTorch 기반으로 구현되었으며, NVIDIA CUDA & NCCL 툴킷을 사용하여 Synchronized Cross-GPU Batch Normalization을 구현하여 GPU 간 배치 크기를 동기화합니다.
  - 학습률은 $lr = \text{baselr} \times (1 - \frac{\text{iter}}{\text{totaliter}})^{\text{power}}$ 스케줄링을 사용하며, 데이터 증강(무작위 뒤집기, 스케일링, 회전, 고정 크롭)을 적용합니다.
  - SE-loss의 ground truth는 세그멘테이션 마스크에서 직접 생성됩니다. 최종 손실은 픽셀 단위 분할 손실과 SE-loss의 가중 합입니다.
  - SE-loss의 가중치 $\alpha$는 경험적으로 $0.2$가 최적이었으며, Encoding Layer의 코드워드 수 $K=32$를 사용했습니다.

## 📊 Results

- **PASCAL-Context 데이터셋:**
  - EncNet은 FCN baseline (73.4% pixAcc, 41.0% mIoU)에 비해 Context Encoding Module만으로도 78.1% pixAcc, 47.6% mIoU로 크게 향상되었습니다.
  - SE-loss를 추가하고 ResNet101을 사용했을 때 80.4% pixAcc, 51.7% mIoU를 달성했습니다.
  - 멀티 스케일 평가 시 81.2% pixAcc, 52.6% mIoU (59 클래스)를 기록했으며, 배경 포함 51.7% mIoU로 RefineNet (47.3%) 등 기존 최첨단 모델들을 능가했습니다.
- **PASCAL VOC 2012 데이터셋:**
  - COCO 사전 학습 없이 82.9% mIoU를 달성하여 기존 모든 방법들을 뛰어넘었습니다.
  - MS-COCO 데이터셋 사전 학습 후 PASCAL 데이터셋에서 미세 조정했을 때 85.9% mIoU를 달성하여 PSPNet [59] 및 DeepLabv3 [6]와 같은 최첨단 모델들을 능가하며, 계산 복잡도 면에서도 효율성을 보였습니다.
- **ADE20K 데이터셋:**
  - EncNet-101은 baseline FCN을 크게 능가하며, 44.65% mIoU를 기록하여 PSPNet-269 (44.94% mIoU)와 비견되는 성능을 보였습니다.
  - ADE20K 테스트 세트에서 EncNet은 0.5567의 최종 점수를 달성, COCO-Place Challenge 2017의 우승작과 2016년 PSP-Net-269의 기록을 넘어섰습니다.
- **CIFAR-10 이미지 분류:**
  - 14개 계층의 얕은 ResNet에 Context Encoding Module을 적용하여 3.45%의 테스트 오차율을 달성했습니다. 이는 ResNet 100층 (4.62%), Wide ResNet (3.89%), DenseNet (3.46%) 등 훨씬 깊거나 파라미터 수가 많은 최첨단 모델들과 비교해도 우수한 성능입니다.
  - Encoding Layer의 스케일링 팩터($s_k$)에 드롭아웃/쉐이크아웃과 유사한 정규화를 적용하여 훈련 최적화에 도움을 주었습니다.

## 🧠 Insights & Discussion

- **명시적 전역 컨텍스트 활용의 효과:** 이 논문은 FCN 기반 semantic segmentation에서 발생하는 픽셀 단위 오분류 문제가 전역 컨텍스트 정보의 부족 때문임을 지적하고, Context Encoding Module을 통해 이를 명시적으로 해결하는 것이 매우 효과적임을 입증했습니다. 이는 단순히 수용 영역을 확대하는 것과는 다른, 의미론적 이해를 통한 성능 향상을 의미합니다.
- **클래스 종속 피처맵 조절의 중요성:** Encoding Layer에서 캡처한 인코딩된 의미론을 바탕으로 피처맵의 스케일링 팩터를 동적으로 예측하여, 특정 장면에 맞는 클래스 관련 피처맵을 강조하거나 약화시키는 전략이 네트워크의 분류를 '간소화'하는 데 핵심적인 역할을 합니다.
- **SE-loss의 보완적 역할:** 픽셀 단위 손실이 놓칠 수 있는 전역적인 객체 존재 여부 정보를 SE-loss를 통해 네트워크에 학습시킴으로써, 특히 작은 객체들의 세그멘테이션 성능을 개선할 수 있었습니다. 이는 손실 함수의 설계가 모델의 컨텍스트 이해도에 큰 영향을 미 미침을 보여줍니다.
- **효율성 및 실용성:** 제안된 Context Encoding Module은 기존 FCN 아키텍처에 경량 모듈로 쉽게 통합될 수 있으며, 미미한 추가 계산 비용으로 상당한 성능 향상을 가져옵니다. 이는 제안 방식이 실용적인 애플리케이션에 적용 가능함을 시사합니다.
- **일반화 가능성:** Semantic Segmentation뿐만 아니라 이미지 분류와 같은 다른 시각 인식 작업에서도 Context Encoding Module이 얕은 네트워크의 특징 표현 능력을 향상시킬 수 있음이 증명되었습니다. 이는 전역 컨텍스트 인코딩 전략이 다양한 딥러닝 모델 및 태스크에 적용될 수 있는 일반성을 가짐을 시사합니다.
- **제한 사항:** SE-loss의 가중치 $\alpha$와 코드워드 수 $K$ 등 일부 하이퍼파라미터는 경험적으로 최적값을 찾았으며, 이는 데이터셋이나 특정 애플리케이션에 따라 튜닝이 필요할 수 있습니다.

## 📌 TL;DR

- **문제:** FCN 기반 Semantic Segmentation은 전역 컨텍스트 부족으로 픽셀 단위 오분류에 취약합니다.
- **제안 방법:** 논문은 장면의 의미론적 컨텍스트를 캡처하여 클래스 종속 피처맵을 선택적으로 강조하는 Context Encoding Module (CEM)과, 전역 컨텍스트 학습을 정규화하는 Semantic Encoding Loss (SE-loss)를 제안합니다. 이를 통합한 Context Encoding Network (EncNet)을 구축했습니다.
- **주요 결과:** EncNet은 PASCAL-Context, PASCAL VOC 2012, ADE20K 등 주요 Semantic Segmentation 벤치마크에서 기존 최첨단 성능을 뛰어넘는 결과를 달성했습니다. 또한, CIFAR-10 이미지 분류 태스크에서 얕은 네트워크의 특징 표현력을 크게 향상시켜, CEM이 적은 추가 계산으로 전역 컨텍스트 활용의 중요성을 입증했습니다.
