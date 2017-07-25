function tts(sentence1,sentence2)
if nargin<1
    sentence1 = 'Speak now';
    sentence2 = 'and the output is :'
end
try
    NET.addAssembly('System.Speech');
    Speaker = System.Speech.Synthesis.SpeechSynthesizer;
    if ~isa(sentence1,'cell')
        sentence = {sentence1};
    end
     for n=1:length(sentence)
        Speaker.Speak (sentence{n});
    end
    
    
    
    if ~isa(sentence2,'cell')
        sentence = {sentence2};
    end
    
    
    for n=1:length(sentence)
        Speaker.Speak (sentence{n});
    end
end

        
