# Edge-Gated CNNs for Volumetric Semantic Segmentation of Medical Images

Ali Hatamizadeh, Demetri Terzopoulos, Andriy Myronenko (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상의 volumetric semantic segmentation에서 발생하는 부정확한 경계 획정(boundary delineation) 문제를 해결하고자 한다. 

일반적인 Convolutional Neural Networks(CNNs)는 이미지 인식 과정에서 형태(shape) 정보보다는 질감(texture) 추상화에 편향되어 학습하는 경향이 있다. 의료 영상 분석에서 장기나 병변의 경계는 진단 및 치료 계획 수립에 매우 중요한 정보이며, 실제로 전문의들은 수동 분할 시 경계를 먼저 식별한 후 내부 영역을 채우는 방식을 사용한다. 그러나 기존의 CNN 아키텍처는 이러한 경계 정보를 충분히 활용하지 못하며, 이로 인해 예측 결과의 경계가 모호해지는 문제가 발생한다.

따라서 본 연구의 목표는 기존의 encoder-decoder 구조에 통합 가능한 plug-and-play 모듈인 Edge-Gated CNNs(EG-CNNs)를 제안하여, 질감 정보와 경계 정보를 동시에 효과적으로 처리함으로써 분할 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 질감(texture) 학습과 경계(edge) 학습을 분리하여 처리하되, 이를 상호 보완적으로 융합하는 것이다.

1. **Edge-Gated Layer 제안**: 메인 네트워크(Main stream)의 여러 해상도에서 추출된 feature map을 입력으로 받아 경계 세만틱을 점진적으로 강조하는 효율적인 edge-gated layer를 설계하였다.
2. **Dual-task Learning 및 Consistency Loss**: 경계 예측과 영역 분할 예측을 별개로 수행하는 것이 아니라, 두 작업 간의 일관성을 강제하는 consistency loss를 도입하여 공동 학습(joint learning)을 수행한다. 이를 통해 추가적인 데이터 어노테이션 비용 없이 경계 인식 능력을 향상시켰다.
3. **Plug-and-play 모듈화**: 특정 아키텍처에 종속되지 않고 U-Net, V-Net, Seg-Net 등 다양한 기존 encoder-decoder 구조에 쉽게 결합하여 성능을 개선할 수 있는 범용적인 구조를 제안하였다.

## 📎 Related Works

기존의 의료 영상 분할 연구들은 주로 U-Net과 같이 down-sampling과 up-sampling 경로를 가진 encoder-decoder 구조를 통해 다중 스케일 특징 표현을 학습하는 방식에 집중해 왔다. 하지만 이러한 접근 방식은 단일 파이프라인 내에서 서로 다른 추상화 수준의 정보를 처리하므로 형태 정보의 손실이 발생하기 쉽다.

이를 보완하기 위해 일부 연구에서는 후처리(post-processing) 과정을 통해 손실된 형태 디테일을 복구하려 했으나, 이는 학습 단계에서 근본적으로 경계 정보를 반영하는 것이 아니라는 한계가 있다. 본 논문은 인간의 시각 시스템이 경계를 먼저 인식하는 방식에서 영감을 얻어, 네트워크 내부에서 명시적으로 경계 정보를 추출하고 이를 분할 결과에 반영하는 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
EG-CNN은 기존의 generic encoder-decoder 네트워크(Main stream)와 병렬로 동작하는 edge stream으로 구성된다. 
1. **Main Stream**: 표준적인 encoder-decoder 구조를 통해 semantic segmentation을 수행한다.
2. **Edge Stream (EG-CNN)**: Main stream의 각 해상도별 feature map을 입력받아 residual block과 edge-gated layer를 거치며 경계 정보를 추출한다.
3. **Fusion**: 최종적으로 EG-CNN의 출력과 Main stream의 출력을 결합(concatenate)하여 최종 분할 맵을 생성한다.

### Edge-Gated Layer
Edge-gated layer는 메인 스트림의 특징 맵과 에지 스트림의 특징 맵을 연결하여 경계 특징을 강조한다. 해상도 $r$에서 에지 스트림의 입력 $e_{r,in}$과 메인 스트림의 입력 $m_r$이 주어졌을 때, 먼저 $1 \times 1 \times 1$ convolution과 ReLU를 통해 attention map $\alpha_r$을 생성한다.

$$\alpha_r = \sigma(\text{Re}(C_{1\times1\times1}(e_{r,in}) + C_{1\times1\times1}(m_r)))$$

여기서 $\sigma$는 sigmoid 함수를 의미한다. 생성된 $\alpha_r$은 $e_{r,in}$과 픽셀 단위로 곱해진 후 residual layer를 통과하여 최종 출력 $e_{r,out}$이 된다.

$$e_{r,out} = e_{r,in} \odot \alpha_r + e_{r,in}$$

### 손실 함수 (Loss Functions)
전체 학습 목표는 다음과 같은 세 가지 손실 함수의 합으로 정의된다.

$$L_{Tot} = L_{Semantic} + L_{Consistency} + L_{Edge}$$

1. **Semantic Loss ($L_{Semantic}$)**: 메인 스트림의 질감 표현 학습을 위해 표준적인 Dice loss를 사용한다.
   $$L_{Dice} = 1 - \frac{2 \sum y_{true} y_{pred}}{\sum y_{true}^2 + \sum y_{pred}^2 + \epsilon}$$

2. **Edge Loss ($L_{Edge}$)**: 경계 표현 학습을 위해 Dice loss와 Balanced Cross Entropy(BCE)를 결합하여 사용한다. 특히 BCE는 경계 픽셀과 비경계 픽셀 간의 불균형을 해결하기 위해 비경계 픽셀의 가중치를 $\beta$로 조절한다.
   $$L_{BCE} = -\beta \sum_{j \in e^+} \log P(e_{pred,j}=1|x;\theta) - (1-\beta) \sum_{j \in e^-} \log P(e_{pred,j}=0|x;\theta)$$

3. **Consistency Loss ($L_{Consistency}$)**: 예측된 semantic mask의 경계와 실제 ground-truth 경계 사이의 불일치를 $L1$ loss로 페널티를 부여한다.
   $$L_{Consistency} = \sum_{j \in e^+} (\|\nabla(\arg \max(P(y_{pred,j}=1|e;c)))\| - \|\nabla(y_{true,j})\|)$$
   이때 $\arg \max$ 함수는 미분 불가능하므로, Gumbel-softmax trick을 사용하여 그래디언트 전파가 가능하도록 근사화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: BraTS 2019 (뇌종양 MRI, 335례), KiTS 2019 (신장 종양 CT, 300례)
- **백본 네트워크**: U-Net, V-Net, Seg-Net
- **평가 지표**: Dice score (BraTS는 WT, TC, ET 각 부위별 측정 / KiTS는 Kidneys, Tumor, Composite Dice 측정)
- **구현**: Pytorch 기반, 8x NVIDIA Tesla V100 GPU 사용, Adam optimizer 적용

### 정량적 결과
- **BraTS 2019**: 모든 백본 네트워크에서 EG-CNN을 추가했을 때 Dice score가 일관되게 상승하였다. 특히 Seg-Net에 EG-CNN을 적용했을 때 Average Dice가 $0.8300$에서 $0.8570$으로 크게 향상되었다.
- **KiTS 2019**: 신장(Kidneys)과 종양(Tumor) 모두에서 성능 향상이 관찰되었으며, Seg-Net+EG-CNN 조합이 가장 높은 Composite Dice($0.9000$)를 기록하였다.

### 정성적 결과
시각화 결과, EG-CNN을 적용한 모델이 단독 모델에 비해 훨씬 더 정교하고 선명한(crisp) 경계를 생성함을 확인하였다. 특히 일반적인 BCE loss 사용 시 발생하는 경계 두꺼워짐 현상이 억제되었으며, 복잡하고 불규칙한 뇌종양의 경계 표현 능력이 크게 개선되었다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 모델의 단순한 용량(capacity) 증가가 아닌, 경계 정보라는 특화된 세만틱을 명시적으로 학습시킴으로써 성능을 높였다는 점이다. 저자들은 네트워크의 깊이나 너비를 늘린 더 큰 standalone 모델들과 비교 실험을 진행하였으나, 단순히 모델 크기를 키우는 것보다 EG-CNN 모듈을 추가하는 것이 검증 정확도 향상에 더 효과적임을 입증하였다.

또한, edge ground-truth를 생성하기 위해 별도의 수동 어노테이션 없이 기존 mask에 3D Sobel 필터를 적용하는 online 방식을 채택하여 실용성을 높였다. 다만, 본 논문에서는 세 가지의 대중적인 아키텍처에 대해서만 검증하였으므로, 더 다양한 최신 아키텍처(예: Transformer 기반 모델)에서의 범용성에 대한 추가 검증이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 의료 영상 분할 시 CNN이 질감 정보에 치중하여 경계 표현이 부정확해지는 문제를 해결하기 위해, 경계 정보를 전담하여 처리하는 **Edge-Gated CNN(EG-CNN)** 모듈을 제안한다. 이 모듈은 메인 네트워크의 특징 맵을 이용해 경계를 강조하는 Edge-gated layer와, 영역-경계 간의 일관성을 강제하는 Consistency loss를 통해 작동한다. 실험 결과, U-Net, V-Net, Seg-Net 등 다양한 백본에서 일관된 성능 향상을 보였으며, 특히 복잡한 해부학적 구조의 경계를 정밀하게 획정하는 데 기여함으로써 향후 정밀 의료 영상 분석 및 진단 시스템에 유용하게 적용될 가능성이 높다.