# Gland Instance Segmentation by Deep Multichannel Neural Networks

Yan Xu, Yang Li, Mingyuan Liu, Yipei Wang, Yubo Fan, Maode Lai, and Eric I-Chao Chang

## 🧩 Problem to Solve

이 논문은 대장 조직학 이미지에서 개별 선(gland)을 인스턴스 분할하는 문제를 다룹니다. 이 작업은 다음과 같은 이유로 매우 어렵습니다:

* **복잡한 배경**: 선을 복잡한 배경으로부터 분리해야 합니다.
* **개별 식별의 어려움**: 같은 클래스에 속하는 선이라 할지라도 개별적으로 식별해야 합니다.
* **다양한 형태 및 이질성**: 선의 형태가 매우 이질적이며, 세포 내외 기질의 변화로 인해 색상 불균일성(anisochromasia)이 발생합니다.
* **선간 인접성 (Coalescence)**: 여러 선이 매우 가깝게 위치하거나 서로 붙어 있어 미세한 간격만 있거나 경계가 서로 접착된 경우 컴퓨터가 이를 하나의 개체로 오인할 수 있습니다.

## ✨ Key Contributions

* **깊은 다중 채널 신경망 (Deep Multichannel Neural Networks) 프레임워크 제안**: 선 인스턴스 분할을 위한 새로운 프레임워크를 개발했습니다. 이 프레임워크는 이미지의 지역(region), 위치(location), 경계(boundary) 패턴과 같은 복잡한 다중 채널 정보를 자동으로 활용하고 융합합니다.
* **다양한 CNN 모델의 통합**: 영역 분할(FCN), 객체 감지(Faster R-CNN), 경계 감지(HED)를 위한 세 가지 채널을 활용하여 상호 보완적인 정보를 제공합니다.
* **최첨단 성능 달성**: MICCAI 2015 Gland Segmentation Challenge 데이터셋에서 기존 참가자 및 다른 최신 인스턴스 분할 방법들에 비해 우수한 결과를 달성했습니다.
* **확장된 데이터 증강 전략**: 회전 불변 이미지 문제를 해결하기 위해 효과적인 새로운 데이터 증강 전략(사인 변환, 핀쿠션 변환, 전단 변환 포함)을 제안했습니다.
* **효과 검증을 위한 제거 연구 (Ablation Experiments)**: 제안된 프레임워크와 각 채널, atrous convolution, 데이터 증강 전략의 효과를 실험적으로 입증했습니다.

## 📎 Related Works

* **인스턴스 분할**: SDS [14], Hypercolumn [15], MCNs [7], DeepMask [16], SharpMask [17], InstanceFCN [18], DCAN [19] 등 객체 감지 또는 영역 제안(proposal) 기반의 방법론들이 언급되었습니다. 이들은 주로 자연 이미지에 적용되었으며, 의료 이미지의 복잡성과 인스턴스 분할 문제에는 한계가 있었습니다.
* **의미론적 분할 (Semantic Segmentation)**: FCN [4], HED [5], U-Net [26], DeepLab [35] 등 FCN 계열 모델들이 픽셀 단위 레이블링에 강점을 보였으나, 동일 클래스 내 개별 객체(인스턴스) 구분에 어려움이 있습니다.
* **객체 감지 (Object Detection)**: R-CNN [27], Fast R-CNN [3], Faster R-CNN [6], DeepMultiBox [28], YOLO [30] 등 딥러닝 기반의 객체 위치 파악 방법들이 참조되었습니다.
* **경계 감지 (Edge Detection)**: HED [5], DeepEdge [31] 등 멀티 스케일 정보를 활용하는 딥러닝 기반 경계 감지 방법들이 언급되었습니다.
* **이전 연구**: 본 연구의 초기 버전은 Xu et al. [34]에서 발표되었습니다. 본 논문에서는 탐지 채널 추가, 성능 개선, 새로운 데이터 증강 전략 도입, 제거 연구 수행 등의 추가 기여를 합니다.

## 🛠️ Methodology

본 논문은 세 가지 독립적인 채널에서 정보를 추출하고, 이를 얕은(shallow) 컨볼루션 신경망을 통해 융합하여 최종 인스턴스 분할 결과를 생성하는 다중 채널 프레임워크를 제안합니다.

1. **프레임워크 개요**:
    * **영역 분할 채널 (Foreground Segmentation Channel)**: 선과 배경을 구분하는 확률 마스크(region information)를 생성합니다.
    * **객체 감지 채널 (Object Detection Channel)**: 각 선의 경계 상자(bounding box)를 감지하여 위치 정보를 제공합니다.
    * **경계 감지 채널 (Edge Detection Channel)**: 선 간의 정밀한 경계를 감지하여 경계 정보를 제공합니다.
    * **융합 네트워크 (Fusion Network)**: 세 채널의 출력을 결합하여 최종 인스턴스 분할을 수행합니다.

2. **각 채널 상세**:
    * **영역 분할 채널**: Atrous convolution을 적용한 FCN-32s [4]의 변형을 사용합니다. `pool4`와 `pool5` 레이어의 스트라이드를 1로 설정하고, atrous convolution을 적용하여 수용장(receptive field)을 넓히면서도 해상도 손실을 줄여 더 정확한 확률 마스크를 얻습니다.
        $$P_{s}(Y_{s}^{*}=k|X;w_{s}) = \mu_{k}(h_{s}(X,w_{s}))$$
        여기서 $X$는 입력 이미지, $w_{s}$는 FCN 네트워크 파라미터, $\mu(\cdot)$는 softmax 함수입니다.
    * **객체 감지 채널**: Faster R-CNN [6]을 사용하여 선의 경계 상자를 예측합니다. 이 경계 상자 정보는 다른 두 채널과 일관된 형태로 변환됩니다. 각 픽셀의 값은 해당 픽셀이 속한 경계 상자의 수와 동일하게 설정됩니다.
        $$P_{d}(X,w_{d}) = \phi(h_{d}(X,w_{d}))$$
        여기서 $w_{d}$는 Faster R-CNN 파라미터, $\phi$는 경계 상자 채우기 연산입니다.
    * **경계 감지 채널**: Holistically-nested Edge Detector (HED) [5]를 기반으로 합니다. 깊은 감독(deep supervision)과 양성-음성 클래스 간 손실 균형을 사용하여 정확하고 선명한 경계를 예측합니다. 최종 출력은 여러 깊은 감독 출력의 가중 합으로 계산됩니다.
        $$P_{e}(Y_{e}^{*}=1|X;w_{e},\alpha) = \sigma\left(\sum_{m=1}^{M} \alpha^{(m)}\cdot h_{e}^{(m)}(X,w_{e})\right)$$
        여기서 $w_{e}$는 HED 파라미터, $\sigma(\cdot)$는 시그모이드 함수, $\alpha$는 가중치 계수입니다.

3. **다중 채널 융합**:
    * 세 채널의 출력을 입력으로 받아 얕은 7-레이어 컨볼루션 신경망을 사용하여 최종 인스턴스 분할 결과를 생성합니다. 이 융합 네트워크에도 atrous convolution을 적용하여 정보 손실을 줄이고 충분히 큰 수용장을 확보합니다.
        $$P_{f}(Y_{f}^{*}=k|P_{s},P_{d},P_{e};w_{f}) = \mu_{k}(h_{f}(P_{s},P_{d},P_{e},w_{f}))$$
        여기서 $w_{f}$는 융합 네트워크 파라미터입니다.

4. **데이터 증강 및 전처리**:
    * 각 채널별 제로 평균(zero mean) 전처리를 수행합니다.
    * 영역 레이블에서 경계 레이블을 생성한 후 팽창(dilation)을 적용합니다.
    * **데이터 증강 전략 I**: 수평 뒤집기, 0°, 90°, 180°, 270° 회전.
    * **데이터 증강 전략 II**: 전략 I에 추가하여 사인 변환(sinusoidal transformation), 핀쿠션 변환(pin cushion transformation), 전단 변환(shear transformation)을 포함합니다. 이는 형태학적 다양성에 대응하고 강건성을 높입니다.

## 📊 Results

* **데이터셋**: MICCAI 2015 Gland Segmentation Challenge Contest [12, 13]의 데이터셋(165개의 대장암 조직학 이미지, 훈련 85개, 테스트 A 60개, 테스트 B 20개)을 사용했습니다.
* **평가 지표**:
  * **F1 Score**: 선 감지 정확도를 평가합니다 ($F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$).
  * **Object-level Dice**: 분할 성능을 평가하며, 개별 인스턴스 단위의 Dice 계수를 사용합니다 ($D(G,S) = \frac{2(|G \cap S|)}{|G|+|S|}$).
  * **Object-level Hausdorff**: 형태 유사성을 평가하며, 개별 인스턴스 단위의 Hausdorff 거리를 사용합니다.
* **주요 결과**:
  * 제안된 프레임워크는 MICCAI 2015 Challenge의 모든 참가자 및 다른 인기 있는 인스턴스 분할 방법들(SDS, Hypercolumn, MNC)과 비교하여 F1 Score, Object-level Dice, Object-level Hausdorff 지표에서 최첨단 성능을 달성했습니다 (Table I, II). 특히 FCN 단독 및 atrous convolution이 적용된 FCN보다 우수한 결과를 보였습니다.
  * 데이터 증강 전략 II는 전략 I보다 FCN 및 atrous convolution 적용 FCN 모두에서 성능을 크게 향상시켰습니다 (Table III).
  * 제거 연구를 통해 모든 세 가지 채널(영역, 경계, 객체 감지)과 atrous convolution의 적용이 성능 향상에 필수적임을 입증했습니다 (Table IV). 경계 팽창(edge dilation) 또한 네트워크 학습의 불균형 문제를 완화하고 정밀도를 높이는 데 기여했습니다.
* **시각적 결과**: 대부분의 인접한 선 구조가 성공적으로 분리되었음을 보여주었습니다 (Fig. 4).
* **제한 사항**: 일부 작거나 혈액으로 채워진 선, 또는 배경과 세포질의 흰색 영역이 혼동되는 경우 (예: Fig. 4의 마지막 행) 모델이 어려움을 겪는 경우가 있었습니다. 테스트 A(정상 이미지 위주)보다 테스트 B(암 이미지 위주)에서 일반적으로 낮은 순위를 기록했는데, 이는 암성 선의 복잡한 형태와 큰 크기 때문입니다.

## 🧠 Insights & Discussion

* **상호 보완적인 정보의 중요성**: 이 연구는 인스턴스 분할 문제, 특히 복잡한 의료 이미지에서, 단순히 픽셀별 분류(영역)만으로는 부족하며 객체의 위치(detection)와 정밀한 경계(edge) 정보가 상호 보완적으로 결합될 때 훨씬 더 강력한 성능을 발휘한다는 것을 명확하게 보여줍니다. FCN의 한계(인접 객체 병합)를 효과적으로 극복했습니다.
* **Atrous Convolution의 효과**: Downsampling으로 인한 해상도 저하와 정보 손실 없이 수용장을 넓히는 atrous convolution의 활용은 분할 정밀도를 높이는 데 결정적인 역할을 했습니다. 이는 특히 공간에 민감한(space-sensitive) 분할 작업에 중요합니다.
* **데이터 증강의 필요성**: 의료 영상 데이터셋의 규모가 제한적인 상황에서, 다양한 형태 변환을 포함하는 정교한 데이터 증강 전략이 모델의 강건성과 일반화 성능을 향상시키는 데 매우 중요함을 확인했습니다.
* **의료 영상 분석에의 기여**: 제안된 프레임워크는 선 형태학적 평가, 암 등급 분류 및 병기 결정과 같은 임상 진단에 필수적인 도구로 활용될 수 있는 잠재력을 가집니다. 특히 인접한 개별 선을 정확히 분할함으로써 정량적 형태학 평가의 기반을 마련했습니다.
* **향후 과제**: 배경과 세포질의 흰색 영역을 더 잘 구분하고, 매우 작거나 불규칙한 선에 대한 감지 및 분할 정확도를 높이는 것이 필요합니다.

## 📌 TL;DR

이 논문은 대장 조직학 이미지에서 개별 선을 정확히 분할하는 인스턴스 분할 문제를 해결하기 위해 **깊은 다중 채널 신경망 프레임워크**를 제안합니다. 이 프레임워크는 **영역 분할 (FCN 기반), 객체 감지 (Faster R-CNN 기반), 경계 감지 (HED 기반)**의 세 가지 채널에서 얻은 정보를 얕은 컨볼루션 네트워크를 통해 융합합니다. Atrous convolution과 효과적인 데이터 증강 전략을 활용하여 MICCAI 2015 Gland Segmentation Challenge 데이터셋에서 **최첨단 성능을 달성**했으며, 특히 인접한 선들을 성공적으로 분리하여 의료 이미지 분석에 중요한 기여를 했습니다.
