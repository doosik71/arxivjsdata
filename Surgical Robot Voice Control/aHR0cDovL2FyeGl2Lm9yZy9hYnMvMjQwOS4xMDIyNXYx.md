# Voice control interface for surgical robot assistants
Ana Davila, Jacinto Colan and Yasuhisa Hasegawa

## 🧩 Problem to Solve
기존의 로봇 보조 최소 침습 수술(RAMIS) 제어 인터페이스(조이스틱, GUI 등)는 외과의에게 상당한 인지적 부담을 주어 수술 효율성을 저해할 수 있습니다. 이는 외과의가 수술의 핵심적인 측면에 집중하는 것을 방해할 수 있습니다. 수술 효율성, 외과의-로봇 협업 능력 향상, 외과의 부담 경감을 위해 자연스럽고 직관적인 음성 제어 인터페이스의 필요성이 제기됩니다.

## ✨ Key Contributions
*   외과의의 음성 명령을 실시간으로 해석하고 실행하여 수술용 조작기(manipulator)를 제어하는 새로운 음성 제어 인터페이스를 제안합니다.
*   최첨단 음성 인식 모델인 Whisper를 ROS (Robot Operating System) 프레임워크 내에 통합하여 시스템의 견고성과 정확성을 높였습니다.
*   제안된 시스템이 높은 음성 인식 정확도와 빠른 추론 속도를 보여주며, 실제 수술 환경에서 음성 제어의 실현 가능성을 입증했습니다.
*   조직 삼각측량(tissue triangulation) 작업을 통해 실제 수술 적용 가능성을 성공적으로 시연했습니다.

## 📎 Related Works
*   **초기 연구:** Allaf et al. [7]은 AESOP 로봇 제어에 음성 및 발 페달 인터페이스를 비교하여 음성 명령이 인지적 부담을 줄일 수 있음을 시사했습니다. Nathan et al. [8]은 AESOP 시스템이 수술 정밀도 향상에 효과적임을 보였습니다. El-Shallaly et al. [9]은 복강경 담낭 절제술에서 음성 인식 인터페이스(VRI)가 수술 시간과 직원 효율성을 개선함을 발견했습니다.
*   **음성 인식 기술:** Zinchenko et al. [10]은 수술 로봇 내시경의 의도적 음성 인식 제어에 대한 연구를 수행했습니다. He et al. [11]은 상업용 음성 인식 인터페이스를 사용한 코 내시경 수술 로봇의 음성 기반 제어 시스템을 설계했습니다.
*   **오프라인 및 클라우드 기반 시스템:** Vosk 및 Kaldi [12]와 같은 오프라인 툴킷이 사용되었고, 최근에는 클라우드 기반 음성 인식 시스템(예: Google Cloud Speech [14], Alexa [15])이 성능 및 확장성 향상 가능성을 보여주었습니다.
*   **다른 제어 인터페이스와의 비교:** Yang et al. [14]은 발 인터페이스와 음성 제어를 비교하여 발 인터페이스가 더 나은 성능을 보였습니다. Elazzazi et al. [15] 및 Paul et al. [16]은 da Vinci 수술 로봇에서 자율 카메라 제어 시스템을 위한 자연어 인터페이스를 탐구했으며, 음성 제어가 수동 제어보다 선호될 수 있음을 보여주었습니다.
*   **딥러닝 활용:** Domínguez-Vidal and Sanfeliu [17]는 협업 객체 운반 작업에서 음성 명령 인식을 위해 CNN을 활용했습니다.

## 🛠️ Methodology
제안된 시스템은 크게 세 가지 모듈로 구성됩니다: 음성 인식 모듈 (SRM), 매핑 모듈 (MM), 그리고 로봇 제어 모듈입니다.

1.  **음성 인식 모듈 (SRM):**
    *   외과의의 음성 입력을 정확하게 캡처하고 해석하는 역할을 합니다.
    *   OpenAI의 최첨단 자동 음성 인식(ASR) 시스템인 Whisper [18]를 핵심으로 사용합니다. Whisper는 대규모 다국어/다중 작업 데이터셋으로 훈련된 Transformer 시퀀스-투-시퀀스 모델이며, 배경 소음과 악센트에 강합니다.
    *   작업 흐름: 고품질 마이크가 내장된 블루투스 헤드셋을 통해 음성 입력 녹음 → 노이즈 감소 및 필터링을 통한 전처리 → Whisper 모델에 입력하여 실시간으로 음성 인식 수행 → 음성 명령 텍스트 변환(transcript) 생성.

2.  **매핑 모듈 (MM):**
    *   변환된 음성 명령을 특정 로봇 동작으로 해석하고 매핑합니다.
    *   조직 조작 작업을 위해 다음 7가지 명령을 정의합니다: "hey robot"(로봇 활성화), "tense"(조직 당기기), "release"(집게 풀기), "pull more"(더 당기기), "pull less"(덜 당기기), "stop"(로봇 정지), "terminate"(로봇 비활성화).
    *   변환된 명령과 각 미리 정의된 명령 간의 단어 오류율 (WER, Word Error Rate)을 계산합니다. WER이 미리 정의된 임계값보다 낮은 명령 중 가장 낮은 WER을 가진 명령을 선택하여 로봇 컨트롤러로 전송합니다.

3.  **로봇 제어 모듈:**
    *   매핑 모듈에서 제공된 명령을 로봇 조작기의 특정 동작으로 변환합니다.
    *   ROS (Robot Operating System) 프레임워크를 활용하여 모듈성 및 모듈 간 상호 연결을 제공합니다.
    *   SRM/MM은 하나의 ROS 노드로 통합되고, 로봇 제어는 독립적인 ROS 노드로 구현됩니다.
    *   ROS 서비스를 통해 명령을 전달하며, 로봇 컨트롤러는 명령을 수신하고 실행 가능성을 확인한 후 RCM (원격 중심 운동) 제약 [19]을 고려하여 경로 계획 및 동작을 실행하는 액션 서버에 전달합니다.

## 📊 Results
*   **실험 설정:** 7-DoF Kinova Gen3 로봇 조작기와 3-DoF OpenRST [20] 수술 도구로 구성된 로봇 시스템을 사용했습니다. 음성 인터페이스는 고품질 오디오 캡처 및 노이즈 캔슬링 기능이 있는 Beats Flex 블루투스 헤드셋을 사용했습니다.
*   **음성 인식 정확도:** 섹션 3.2에서 정의된 7가지 명령에 대해 인식 정확도를 평가했습니다. 영어가 모국어가 아닌 두 명의 피험자가 각 명령을 30번씩 반복했습니다. 대부분의 명령에서 높은 인식 성능을 보였습니다 (예: "hey robot" 96.7%, "tense" 93.3%, "stop" 100%).
*   **추론 시간:** 사용자가 음성 요청을 완료한 시점부터 로봇 동작이 시작될 때까지의 시간을 측정했습니다. 7가지 음성 명령에 대한 평균 추론 시간은 약 1.7초였습니다. 이는 일반적인 상위 수준 음성 요청에 충분한 시간입니다. 가장 긴 시간은 "pull less" (2.50초)와 "pull more" (2.39초)였습니다.
*   **조직 삼각측량 작업 시연:** 제안된 프레임워크의 실현 가능성을 조직 삼각측량 작업을 통해 시연했습니다. 피험자는 기존 수술 도구를 조작하고 로봇 시스템은 다중 DoF 로봇 수술 도구를 조작했습니다. 작업자는 음성 명령을 사용하여 로봇을 활성화하고 조직에 장력을 가하며 이완하는 등의 작업을 수행했습니다.

## 🧠 Insights & Discussion
*   **중요성:** Whisper 모델과 ROS 프레임워크를 통합한 음성 제어 로봇 보조 시스템은 수술 정밀도와 효율성을 향상시킬 잠재력이 큽니다. 높은 인식 정확도와 안정적인 성능은 실제 수술 환경에서의 실현 가능성을 입증했습니다.
*   **모듈성 및 확장성:** ROS 프레임워크를 활용한 모듈식 아키텍처는 시스템의 유연성과 확장성을 보장하여 다양한 수술 시나리오에 적용될 수 있습니다.
*   **개선점 및 향후 과제:**
    *   **개인화된 훈련:** 외과의의 특정 음성 패턴을 구별하도록 사전 훈련된 모델을 미세 조정(fine-tuning)함으로써 인식 정확도와 신뢰성을 더욱 향상시킬 수 있습니다.
    *   **강건성:** 시끄러운 수술 환경에서의 강건성을 높이는 연구가 필요합니다.
    *   **통합:** 기존 수술 워크플로우와의 통합 및 인공지능, 머신러닝과 같은 다른 고급 기술과의 결합은 수술 로봇 시스템의 효율성과 효과를 크게 향상시킬 수 있습니다.

## 📌 TL;DR
수술 로봇 제어의 인지적 부담을 줄이기 위해, 본 논문은 Whisper 음성 인식 모델과 ROS 프레임워크를 통합한 새로운 음성 제어 인터페이스를 제안합니다. 이 시스템은 외과의의 음성 명령을 실시간으로 인식하고 로봇 동작으로 매핑하여 실행합니다. 실험 결과, 높은 음성 인식 정확도와 평균 1.7초의 추론 속도를 보였으며, 조직 삼각측량 작업 시연을 통해 수술 환경에서의 실현 가능성을 입증했습니다. 이는 수술 효율성과 정밀도 향상에 기여할 잠재력을 보여줍니다.