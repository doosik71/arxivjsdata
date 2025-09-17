# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam

## 🧩 Problem to Solve

이 논문은 시맨틱 이미지 분할(Semantic Image Segmentation)에서 두 가지 주요 접근 방식의 장점을 결합하는 문제를 해결합니다. 기존 방법들은 다음과 같은 한계를 가졌습니다:

1. **공간 피라미드 풀링(Spatial Pyramid Pooling) 또는 ASPP(Atrous Spatial Pyramid Pooling) 기반 네트워크:** 다양한 스케일의 맥락 정보(contextual information)를 효과적으로 인코딩하지만, 풀링 또는 스트라이드(striding) 연산으로 인해 객체 경계의 세부 정보를 손실하여 분할 결과가 뭉개지는 경향이 있습니다.
2. **인코더-디코더(Encoder-Decoder) 구조:** 공간 정보를 점진적으로 복구하여 더 선명한 객체 경계를 얻을 수 있지만, 다중 스케일 맥락 정보 캡처에 한계가 있을 수 있습니다.
   따라서, 풍부한 맥락 정보를 인코딩하면서 동시에 정확한 객체 경계를 복구하는 효율적인 네트워크 구조를 제안하는 것이 목표입니다.

## ✨ Key Contributions

- **새로운 인코더-디코더 구조 제안:** 강력한 인코더 모듈로 DeepLabv3를 사용하고, 간단하지만 효과적인 디코더 모듈을 추가하여 객체 경계 세분화를 개선했습니다.
- **자유로운 인코더 특징 해상도 제어:** Atrous Convolution을 통해 추출된 인코더 특징의 해상도를 임의로 조절하여 정확도와 런타임 간의 균형을 조절할 수 있도록 했습니다. 이는 기존 인코더-디코더 모델에서는 불가능했던 부분입니다.
- **Xception 모델 및 Atrous Separable Convolution 적용:** 시맨틱 분할 작업을 위해 Xception 모델을 개조하고, ASPP 모듈과 디코더 모듈 모두에 Depthwise Separable Convolution을 적용하여 더 빠르고 강력한 인코더-디코더 네트워크를 구축했습니다.
- **최고 성능 달성:** PASCAL VOC 2012 및 Cityscapes 데이터셋에서 새로운 최첨단(state-of-the-art) 성능을 달성했습니다.
- **공개된 구현:** 제안된 모델의 TensorFlow 기반 구현을 공개하여 연구 커뮤니티에 기여했습니다.

## 📎 Related Works

- **FCN (Fully Convolutional Networks) 기반 모델 [8, 11]:** 시맨틱 분할 벤치마크에서 상당한 개선을 보여주며 이 연구의 기반이 되었습니다.
- **공간 피라미드 풀링 (Spatial Pyramid Pooling):**
  - **PSPNet [24]**: 다양한 그리드 스케일에서 풀링 연산을 수행하여 다중 스케일 정보를 활용합니다.
  - **DeepLab [39, 23]**: Atrous Spatial Pyramid Pooling (ASPP)을 사용하여 다양한 비율의 Atrous Convolution을 병렬로 적용합니다.
- **인코더-디코더 (Encoder-Decoder) 네트워크:**
  - **U-Net [21] 및 SegNet [22]**: 특징 맵을 점진적으로 줄여 고수준의 의미 정보를 캡처하는 인코더와 공간 정보를 점진적으로 복구하는 디코더로 구성됩니다.
- **Depthwise Separable Convolution [27, 28]:** 계산 비용과 파라미터 수를 크게 줄이면서도 비슷한 성능을 유지하는 효율적인 연산입니다. Xception [26] 등 많은 최신 신경망 설계에 채택되었습니다.

## 🛠️ Methodology

제안하는 DeepLabv3+ 모델은 DeepLabv3를 강력한 인코더로 사용하고, 그 위에 효율적인 디코더 모듈을 추가하여 객체 경계를 정교하게 만듭니다.

1. **Atrous Convolution (Dilated Convolution):**
   - 일반적인 컨볼루션 연산을 일반화한 것으로, `rate` 매개변수를 통해 입력 신호를 샘플링하는 보폭을 제어합니다.
   - 이를 통해 필터의 수용 영역(field-of-view)을 적응적으로 조절하여 다중 스케일 정보를 캡처하고, 동시에 특징 맵의 해상도를 제어할 수 있습니다.
   - 수식: $y[i] = \sum_k x[i+r \cdot k]w[k]$, 여기서 $r$은 Atrous rate.
2. **Depthwise Separable Convolution:**
   - 표준 컨볼루션을 Depthwise Convolution과 Pointwise Convolution (1x1 Convolution)으로 분리합니다.
   - **Depthwise Convolution:** 각 입력 채널에 대해 독립적으로 공간 컨볼루션을 수행합니다.
   - **Pointwise Convolution:** Depthwise Convolution의 출력을 채널에 걸쳐 결합합니다.
   - **Atrous Separable Convolution:** Depthwise Convolution에 Atrous Convolution을 적용하여 계산 복잡도를 크게 줄이면서 성능을 유지하거나 향상시킵니다.
3. **DeepLabv3를 인코더로 활용:**
   - DeepLabv3는 Atrous Convolution을 사용하여 임의의 해상도(`output_stride`)로 특징을 추출합니다 (예: $output\_stride=16$ 또는 $8$).
   - Atrous Spatial Pyramid Pooling (ASPP) 모듈과 이미지 레벨 특징(image-level features)을 결합하여 다중 스케일 컨텍스트 정보를 인코딩합니다.
   - 인코더 출력 특징 맵은 풍부한 의미론적 정보를 담고 있습니다 (256 채널).
4. **제안하는 디코더 모듈:**
   - 인코더 특징(예: $output\_stride=16$으로 계산된 특징)을 먼저 4배 이중선형 업샘플링(bilinear upsampling)합니다.
   - 이 특징을 네트워크 백본의 해당 저수준 특징(low-level features, 예: ResNet-101의 Conv2)과 연결(concatenate)합니다.
   - 저수준 특징의 채널 수를 줄이기 위해 $1 \times 1$ 컨볼루션을 적용합니다 (예: 48개 채널로). 이는 고수준 특징의 중요성을 압도하지 않도록 합니다.
   - 연결된 특징에 몇 개의 $3 \times 3$ 컨볼루션 (예: 두 개의 $3 \times 3$ 컨볼루션, 각 256 필터)을 적용하여 특징을 정교하게 만듭니다.
   - 마지막으로 4배 이중선형 업샘플링을 다시 수행하여 최종 분할 결과를 얻습니다 (최종 $output\_stride=4$).
5. **수정된 Aligned Xception 백본:**
   - Xception 모델을 시맨틱 분할 작업에 맞게 수정했습니다.
   - 모든 최대 풀링(max pooling) 연산을 스트라이딩을 포함하는 Depthwise Separable Convolution으로 대체하여 Atrous Separable Convolution을 적용할 수 있도록 했습니다.
   - 각 $3 \times 3$ Depthwise Convolution 후에 추가 배치 정규화(Batch Normalization) 및 ReLU 활성화 함수를 추가하여 MobileNet 디자인과 유사하게 개선했습니다.

## 📊 Results

- **PASCAL VOC 2012 데이터셋:**
  - **ResNet-101 백본 사용 시:**
    - 제안된 디코더 모듈을 추가함으로써 mIOU (mean Intersection-over-Union)가 크게 향상됩니다 (예: $output\_stride=16$에서 77.21% $\rightarrow$ 78.85%).
    - Multi-scale 입력 및 좌우 반전(Flip)을 함께 사용하면 성능이 더욱 향상됩니다.
  - **수정된 Xception 백본 사용 시:**
    - DeepLabv3+ (Xception) 모델은 PASCAL VOC 2012 테스트 세트에서 87.8% (COCO 사전 학습), 89.0% (JFT 사전 학습 포함)의 mIOU를 달성하여 새로운 최첨단 성능을 기록했습니다.
    - Depthwise Separable Convolution을 ASPP 및 디코더 모듈에 적용함으로써 mIOU는 비슷하게 유지되면서도 Multiply-Adds 계산 복잡도가 33% $\sim$ 41% 크게 감소하여 효율성이 향상되었습니다.
- **Cityscapes 데이터셋:**
  - DeepLabv3+ (Xception-71) 모델은 Cityscapes 테스트 세트에서 82.1%의 mIOU를 달성하여 역시 새로운 최첨단 성능을 기록했습니다.
  - 디코더 모듈의 추가로 검증 세트에서 1.46%의 성능 향상이 있었습니다.
- **객체 경계 개선 (Trimap Experiment):**
  - 객체 경계 주변의 `void` 픽셀에 대한 `trimap` 분석 결과, 제안된 디코더 모듈은 ResNet-101 및 Xception 백본 모두에서 경계 부근의 mIOU를 크게 향상시켰습니다. 특히 좁은 `trimap` 폭에서 ResNet-101은 4.8%, Xception은 5.4%의 mIOU 향상을 보였습니다.
- **정성적 결과 및 한계:**
  - 모델은 복잡한 객체도 잘 분할하지만, 소파와 의자 구분, 심하게 가려진 객체, 드문 시점의 객체에서는 어려움을 겪는 한계를 보였습니다.

## 🧠 Insights & Discussion

- **인코더-디코더 결합의 시너지:** DeepLabv3의 강력한 다중 스케일 컨텍스트 인코딩 능력과 디코더의 섬세한 경계 복구 능력이 결합되어 시맨틱 분할 성능을 크게 향상시켰습니다. 이는 고수준의 의미론적 정보와 저수준의 공간적 세부 정보 간의 효과적인 결합의 중요성을 보여줍니다.
- **Atrous Convolution의 유연성:** Atrous Convolution을 통해 `output_stride`를 조절하여 계산 자원 예산에 따라 정확도와 속도 간의 균형을 자유롭게 조절할 수 있는 유연성을 제공합니다.
- **Depthwise Separable Convolution의 효율성:** Xception 모델과 Atrous Separable Convolution의 채택은 모델을 더 빠르고 강력하게 만들면서도, 특히 모바일/엣지 디바이스와 같은 자원 제약이 있는 환경에 대한 실용성을 높였습니다.
- **한계:** 특정 종류의 객체(예: 모양이 유사한 소파/의자), 심하게 가려진 객체, 또는 학습 데이터에서 드물게 나타나는 시점의 객체에 대한 분할 오류는 여전히 개선이 필요한 부분입니다. 이는 모델이 아직 충분히 강건하지 않거나, 이러한 특정 경우에 대한 추가적인 컨텍스트 또는 표현 학습이 필요함을 시사합니다.

## 📌 TL;DR

이 논문은 시맨틱 분할에서 맥락 정보와 정교한 경계 복구의 균형을 맞추기 위해 DeepLabv3 인코더와 새로운 디코더 모듈을 결합한 **DeepLabv3+**를 제안합니다. Atrous Convolution을 통해 특징 해상도를 유연하게 제어하고, Xception 백본 및 Atrous Separable Convolution을 도입하여 효율성과 성능을 동시에 개선했습니다. 결과적으로 PASCAL VOC 2012와 Cityscapes에서 최고 수준의 정확도를 달성하며, 효율적이면서도 강력한 시맨틱 분할 모델임을 입증했습니다.
