# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han

## 🧩 Problem to Solve

고해상도 밀집 예측(dense prediction)은 자율주행, 의료 영상 처리, 컴퓨터 사진 등 다양한 실제 애플리케이션에 필수적이지만, 최첨단(SOTA) 모델의 막대한 계산 비용으로 인해 하드웨어 장치에 배포하기가 어렵습니다. 기존 SOTA 모델들은 높은 성능을 위해 컴퓨팅 복잡도가 높은 소프트맥스 어텐션, 하드웨어 비효율적인 대형 커널 컨볼루션, 또는 복잡한 토폴로지 구조에 의존합니다. 이는 모델 배포를 비실용적으로 만듭니다.

## ✨ Key Contributions

* **새로운 멀티스케일 선형 어텐션 모듈 제안:** 효율적인 고해상도 밀집 예측을 위해 전역적 수용장(global receptive field)과 멀티스케일 학습(multi-scale learning)을 하드웨어 효율적인 방식으로 동시에 달성합니다. 고해상도 밀집 예측에 선형 어텐션의 효과를 입증한 첫 번째 연구입니다.
* **EfficientViT 아키텍처 설계:** 제안된 멀티스케일 선형 어텐션 모듈을 기반으로 하는 새로운 고해상도 비전 모델 패밀리인 EfficientViT를 설계했습니다.
* **다양한 하드웨어 플랫폼에서의 탁월한 성능 개선:** 의미론적 분할(semantic segmentation), 초해상도(super-resolution), Segment Anything, ImageNet 분류 등 다양한 작업에서 모바일 CPU, 엣지 GPU, 클라우드 GPU를 포함한 여러 하드웨어 플랫폼에서 기존 SOTA 모델 대비 상당한 속도 향상을 보여주면서도 동일하거나 더 높은 성능을 달성합니다.

## 📎 Related Works

* **고해상도 밀집 예측:** SegNet, U-Net, DeepLab, PSPNet, OCRNet, HRNet 등 CNN 기반의 성능 개선 연구 및 효율성 개선 연구(ICNet, Fast-SCNN, DFANet, BiSeNet)가 있었으나, 효율성과 SOTA 성능 간에 큰 격차가 존재했습니다.
* **효율적인 비전 트랜스포머 (ViT):** MobileViT, MobileFormer, NASViT 등 ViT의 효율성을 개선하려는 시도가 있었으나, 주로 이미지 분류에 초점을 맞췄고 여전히 이차 복잡도의 소프트맥스 어텐션에 의존하여 고해상도 밀집 예측에는 부적합했습니다.
* **효율적인 딥러닝:** 네트워크 가지치기(pruning), 양자화(quantization), 효율적인 모델 아키텍처 설계(depthwise separable convolution, ShuffleNet V2), 훈련 기법(지식 증류, 네트워크 증강), AutoML(NAS, AMC, APQ) 등 다양한 기술이 활용되었습니다.
* **벤치마크 비교 모델:** SegFormer, SegNeXt (의미론적 분할), Restormer, SwinIR, VapSR, BSRN (초해상도), SAM-ViT-H (Segment Anything), CoAtNet, ConvNeXt, Swin, EfficientNetV2, FasterViT, MobileViTV2 (ImageNet 분류 백본).

## 🛠️ Methodology

EfficientViT의 핵심은 하드웨어 효율적인 연산을 통해 전역적 수용장과 멀티스케일 학습을 동시에 가능하게 하는 새로운 **멀티스케일 선형 어텐션 모듈**입니다.

1. **전역적 수용장을 위한 ReLU 선형 어텐션 활용:**
    * 기존 소프트맥스 어텐션($\text{Sim}(Q, K) = \exp(\frac{QK^T}{\sqrt{d}})$) 대신 **ReLU 선형 어텐션**($\text{Sim}(Q, K) = \text{ReLU}(Q)\text{ReLU}(K)^T$)을 사용합니다.
    * 행렬 곱셈의 결합 법칙을 활용하여 계산 복잡도와 메모리 사용량을 입력 해상도에 대해 **이차($O(N^2)$)에서 선형($O(N)$)으로 감소**시킵니다.
    * $$O_i = \frac{\text{ReLU}(Q_i)(\sum_{j=1}^{N} \text{ReLU}(K_j)^T V_j)}{\text{ReLU}(Q_i)(\sum_{j=1}^{N} \text{ReLU}(K_j)^T)}$$
    * 소프트맥스와 같은 하드웨어 비친화적인 연산을 제거하여 모바일 CPU에서 3.3배에서 4.5배 더 빠릅니다.

2. **ReLU 선형 어텐션의 한계 극복:**
    * ReLU 선형 어텐션은 비선형 유사성 함수가 없어 집중된 어텐션 맵을 생성하기 어렵고, 이로 인해 지역 정보 추출 및 멀티스케일 학습 능력이 제한됩니다.
    * **FFN 레이어에 깊이별 컨볼루션(DWConv) 삽입:** 지역 정보 추출 능력을 향상시킵니다. EfficientViT의 빌딩 블록은 멀티스케일 선형 어텐션(컨텍스트 정보)과 FFN+DWConv(지역 정보)로 구성됩니다.
    * **멀티스케일 토큰 생성을 위한 Q/K/V 토큰 집합:** 작은 커널의 깊이별 분리 가능 컨볼루션(DSConv)을 사용하여 인접한 Q/K/V 토큰의 정보를 집합하여 멀티스케일 토큰을 생성합니다. 이는 그룹 컨볼루션을 통해 효율적으로 구현됩니다. 생성된 멀티스케일 토큰에 ReLU 선형 어텐션을 적용하여 멀티스케일 전역 특성을 추출하고, 최종적으로 선형 투사 레이어를 통해 특성을 융합합니다.

3. **EfficientViT 아키텍처:**
    * 표준 백본-헤드/인코더-디코더 설계를 따릅니다.
    * **백본:** 입력 스템과 4개의 스테이지로 구성되며, 스테이지 3과 4에 EfficientViT 모듈을 삽입합니다. 다운샘플링에는 MBConv를 사용합니다.
    * **헤드:** P2, P3, P4 스테이지의 출력 특성 맵을 1x1 컨볼루션과 업샘플링을 통해 공간 및 채널 크기를 맞춘 후 덧셈으로 융합합니다. 강력한 백본 덕분에 MBConv 블록과 출력 레이어로 구성된 간단한 헤드 디자인을 채택합니다.
    * 다양한 효율성 제약 조건을 만족시키기 위해 EfficientViT-B0~B3 및 EfficientViT-L 시리즈 모델을 설계했습니다.

## 📊 Results

* **ImageNet 분류:** EfficientViT-L2-r384는 86.0%의 Top1 정확도를 달성하며 EfficientNetV2-L보다 0.3% 향상된 정확도를 제공하고 A100 GPU에서 2.6배 빠른 속도를 보여줍니다.
* **의미론적 분할 (Cityscapes, ADE20K):**
  * **Cityscapes:** SegFormer 대비 최대 13배 MACs 절감, 엣지 GPU에서 최대 8.8배 지연 시간 감소 (더 높은 mIoU). SegNeXt 대비 2.0배 MACs 감소, 엣지 GPU에서 3.8배 속도 향상 (더 높은 mIoU). A100 GPU에서 SegNeXt보다 최대 3.9배, SegFormer보다 10.2배 높은 처리량.
  * **ADE20K:** EfficientViT-B1은 SegFormer-B1 대비 5.2배 MACs 감소, 3.5배 GPU 지연 시간 감소와 함께 +0.6 mIoU 향상.
* **초해상도 (FFHQ, BSD100):**
  * **경량 SR (BSD100):** 기존 CNN 기반 SR 대비 0.09dB PSNR 이득 및 유사/낮은 GPU 지연 시간. 기존 ViT 기반 SR(SwinIR, Restormer) 대비 동일 PSNR에서 최대 5.4배 속도 향상.
  * **고해상도 SR (FFHQ):** Restormer 대비 0.11dB PSNR 이득과 함께 최대 6.4배 속도 향상.
* **Segment Anything (SAM):**
  * SAM의 이미지 인코더를 EfficientViT로 대체한 EfficientViT-SAM은 A100 GPU에서 SAM-ViT-H 대비 48.9배 높은 처리량을 달성하며 비슷한 제로샷 인스턴스 분할 성능을 보입니다.
  * EfficientViT-SAM-XL1은 SAM-ViT-H보다 COCO 및 LVIS에서 제로샷 인스턴스 분할 성능이 뛰어나며 A100 GPU에서 16.5배 높은 처리량을 제공합니다.
  * 포인트 프롬프트 분할에서도 대부분의 경우 SAM-ViT-H를 능가합니다 (특히 여러 포인트가 주어졌을 때).

## 🧠 Insights & Discussion

* EfficientViT는 고해상도 밀집 예측에서 성능에 필수적인 전역적 수용장과 멀티스케일 학습을 하드웨어 효율적인 방식으로 구현함으로써, FLOPs 감소가 실제 하드웨어 지연 시간 감소로 이어지는 것을 성공적으로 보여줍니다.
* 기존 소프트맥스 어텐션의 고질적인 계산 비용 문제를 선형 어텐션으로 해결하면서도, 선형 어텐션의 단점(지역 정보 추출 능력 부족)을 깊이별 컨볼루션과 멀티스케일 토큰 집합을 통해 보완하여 성능 저하 없이 효율성을 극대화했습니다.
* 특히 Segment Anything (SAM)과 같은 새로운 작업에 EfficientViT를 적용하여, SAM-ViT-H의 제로샷 성능을 능가하면서도 전례 없는 처리량 향상을 달성하며 성능-효율성 트레이드오프에서 SOTA를 기록했습니다.
* 한계점으로는 단일 포인트 프롬프트 설정에서 성능 개선의 여지가 있으며, 이는 대화형 분할 설정의 부재 때문일 수 있습니다.
* 향후 연구에서는 EfficientViT를 다른 비전 작업에 적용하고 모델을 더욱 확장하는 방안을 모색할 예정입니다.

## 📌 TL;DR

EfficientViT는 고해상도 밀집 예측 모델의 하드웨어 배포 어려움을 해결하기 위해, 하드웨어 친화적인 ReLU 선형 어텐션과 컨볼루션을 결합한 **멀티스케일 선형 어텐션 모듈**을 제안합니다. 이 모듈은 전역적 수용장과 멀티스케일 학습을 효율적으로 구현하여, 기존 SOTA 모델 대비 성능 손실 없이 다양한 하드웨어 플랫폼(모바일 CPU, 엣지/클라우드 GPU)에서 **상당한 속도 향상**을 달성했습니다. 특히 Segment Anything 작업에서 SAM-ViT-H를 능가하는 성능과 48.9배의 처리량 향상을 보여주며 탁월한 성능-효율성 트레이드오프를 입증했습니다.
