# Error examples (selected automatically)


## Seq2Seq + LSTM


### repeated

- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም አንድ ወንድም ወይም አንድ ወንድ ልጅ ወይም ድምጻቸውን ለመንከባከብ ወይም በትዳራቸውን ለመንከባከብ ጥረት ማድረግ ይችላሉ።
- **SRC:** these include the promised " new heavens and a new earth . "  
  **REF:** ይህም ቃል የተገባልንን " አዲስ ሰማይና አዲስ ምድር " ይጨምራል።  
  **HYP:** እነዚህ ቃላት " አዲስ ቃል ኪዳን " እና " አዲስ ኪዳን " በማለት ይናገራል።
- **SRC:** my wife and i look at each other .  
  **REF:** እኔና ባለቤቴ ተያየን።  
  **HYP:** እኔና ባለቤቴ እኔና ባለቤቴ እርስ በርስ እያገለገልኩ ነው።

### missing

- **SRC:** the dread of you makes my body tremble ;  
  **REF:** አንተን እጅግ ከመፍራቴ የተነሳ ሰውነቴ ይንቀጠቀጣል፤  
  **HYP:** ድምጼን ወስዳለሁ፤
- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** ዶክተር ዊልያም
- **SRC:** " he ' s in heaven , " the commander in chief told the family of one fallen marine in a private moment .  
  **REF:** አንድ ከፍተኛ የሰራዊት አዛዥ በውጊያ ላይ ከሞተ አንድ የባህር ሀይል አባል ቤተሰብ ጋር ሲነጋገሩ " ገነት ገብቷል " ሲሉ መደመጣቸው ተሰማ።  
  **HYP:** " በሰማይ የሚኖረውን ቤተ ክርስቲያን [በገጽ 9 ላይ የሚገኝ ስእል]

### additional

- **SRC:** these include the promised " new heavens and a new earth . "  
  **REF:** ይህም ቃል የተገባልንን " አዲስ ሰማይና አዲስ ምድር " ይጨምራል።  
  **HYP:** እነዚህ ቃላት " አዲስ ቃል ኪዳን " እና " አዲስ ኪዳን " በማለት ይናገራል።
- **SRC:** obviously , the world ' s religions have not been immune from satan ' s influence .  
  **REF:** የአለም ሀይማኖቶች ከሰይጣን ተጽእኖ ነጻ አይደሉም።  
  **HYP:** በአለም ዙሪያ ያሉ ሀይማኖቶች በአለም ላይ ተጽእኖ የሚያሳድሩት ለምንድን ነው?
- **SRC:** my wife and i look at each other .  
  **REF:** እኔና ባለቤቴ ተያየን።  
  **HYP:** እኔና ባለቤቴ እኔና ባለቤቴ እርስ በርስ እያገለገልኩ ነው።

### word order

- **SRC:** hence , none of his servants need to be reluctant to accept help from those who are moved by jehovah to give such assistance .  
  **REF:** ይሆዋ አንዳንዶች ለሌሎች እርዳታ እንዲሰጡ ሊያነሳሳቸው ይችላል፤ በመሆኑም ከአገልጋዮቹ መካከል ማናቸውም ቢሆኑ እንዲህ ያለውን ድጋፍ ለመቀበል ማቅማማት የለባቸውም።  
  **HYP:** በመሆኑም ይሆዋ አገልጋዮቹን ለመርዳት የሚያስችላቸውን እርዳታ ለማግኘት ፈቃደኛ መሆን አለበት።
- **SRC:** many who have had the privilege of conducting progressive bible studies will tell you that few things are more rewarding .  
  **REF:** ጥሩ እድገት የሚያደርጉ የመጽሀፍ ቅዱስ ጥናቶችን የመምራት አጋጣሚ ያገኙ በርካታ ወንድሞችና እህቶች በዚህ ሀሳብ ይስማማሉ።  
  **HYP:** በርካታ የመጽሀፍ ቅዱስ ጥናቶችን በማገልገሉ ረገድ ብዙ ጊዜ ምኞታቸው በጣም አስደሳች ነው።
- **SRC:** fact : a teenager is less likely to rebel when parents set reasonable rules and discuss them with him .  
  **REF:** የተሳሳተ አመለካከት፦ ሁሉም ልጅ ሲጎረምስ መመሪያ አያከብርም፤ ይህ በጉርምስና እድሜ ውስጥ የተለመደ ነገር ነው።  
  **HYP:** እርግጥ ነው፦ ወላጆች፣ ወላጆች፣ ወላጆቻቸውን በተመለከተ ምን አይነት አመለካከት ሊኖራቸው ይገባል?

### named entity

- **SRC:** gaffar , who was born in turkey , was disturbed by the idea of a vengeful god , as taught by his religion .  
  **REF:** በቱርክ የተወለደው ጃፋር ሀይማኖቱ በሚያስተምረው ' አምላክ ተበቃይ ነው ' በሚለው ትምህርት ይረበሽ ነበር።  
  **HYP:** በ1940ዎቹ አመታት፣ ምስጢር፣ የስፔንን ንዴትና ምኞቱ በወንጌላዊነት ህይወት ላይ ተገኝቶ ነበር።
- **SRC:** obviously , the world ' s religions have not been immune from satan ' s influence .  
  **REF:** የአለም ሀይማኖቶች ከሰይጣን ተጽእኖ ነጻ አይደሉም።  
  **HYP:** በአለም ዙሪያ ያሉ ሀይማኖቶች በአለም ላይ ተጽእኖ የሚያሳድሩት ለምንድን ነው?
- **SRC:** and moses and aaron went and gathered together all the elders of the children of israel :  
  **REF:** ሙሴና አሮንም ሄዱ የእስራኤልንም ልጆች ሽማግሌዎች ሁሉ ሰበሰቡ።  
  **HYP:** ሙሴም አሮንንና አሮንን እንዲህ አለው፥

### rare

- **SRC:** besides , they claimed that they had neither the facilities nor the manpower to provide an alternative physical education program .  
  **REF:** ከዚህም በላይ አማራጭ የአካል ማጎልመሻ ትምህርት ለማዘጋጀት የሚረዱ መሳሪያዎችም ሆኑ የሰው ሀይል የለንም አለ።  
  **HYP:** ከዚህም በላይ ቲኦክራሲያዊ ትምህርት ቤት ውስጥ የሚጫወቱት በ1940ዎቹ አመታት 2002 አ. ም.
- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** ዶክተር ዊልያም
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም አንድ ወንድም ወይም አንድ ወንድ ልጅ ወይም ድምጻቸውን ለመንከባከብ ወይም በትዳራቸውን ለመንከባከብ ጥረት ማድረግ ይችላሉ።

## Attention Seq2Seq + LSTM


### repeated

- **SRC:** " he ' s in heaven , " the commander in chief told the family of one fallen marine in a private moment .  
  **REF:** አንድ ከፍተኛ የሰራዊት አዛዥ በውጊያ ላይ ከሞተ አንድ የባህር ሀይል አባል ቤተሰብ ጋር ሲነጋገሩ " ገነት ገብቷል " ሲሉ መደመጣቸው ተሰማ።  
  **HYP:** አንድ ሰው " በሰማይ ያለው በሰማይ በሰማይ ነው " በማለት ተናግሯል።
- **SRC:** in recent decades , batik has gained greater popularity and has become a symbol of indonesian national identity .  
  **REF:** ብዙዎቹ የኢንዶኔዥያ ግዛቶች የራሳቸው የሆነ የአቀላለምና ንድፍ የማውጣት ዘዴ አላቸው።  
  **HYP:** ከቅርብ አስርተ አስርተ አመታት ወዲህ የሲና ብሄራዊ ስም ከፍተኛ ዋጋ አለው።
- **SRC:** at that hananiah the prophet took the yoke bar off the neck of the prophet jeremiah and broke it .  
  **REF:** በዚህ ጊዜ ነቢዩ ሀናንያህ ቀንበሩን ከነቢዩ ኤርምያስ አንገት ላይ ወስዶ ሰበረው።  
  **HYP:** በዚህ ጊዜ ነቢዩና ነቢዩ ነቢዩ ኤርምያስን የነቢዩ ኤርምያስን ረድተውታል።

### missing

- **SRC:** the dread of you makes my body tremble ;  
  **REF:** አንተን እጅግ ከመፍራቴ የተነሳ ሰውነቴ ይንቀጠቀጣል፤  
  **HYP:** የቅርጫት ሽልማቴን ትጠብቃለህ፤
- **SRC:** " he ' s in heaven , " the commander in chief told the family of one fallen marine in a private moment .  
  **REF:** አንድ ከፍተኛ የሰራዊት አዛዥ በውጊያ ላይ ከሞተ አንድ የባህር ሀይል አባል ቤተሰብ ጋር ሲነጋገሩ " ገነት ገብቷል " ሲሉ መደመጣቸው ተሰማ።  
  **HYP:** አንድ ሰው " በሰማይ ያለው በሰማይ በሰማይ ነው " በማለት ተናግሯል።
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም ደግሞ አንድ ወጣት ወንድም ወይም እህትን ለማግኘት ሞክሩ።

### additional

- **SRC:** besides , they claimed that they had neither the facilities nor the manpower to provide an alternative physical education program .  
  **REF:** ከዚህም በላይ አማራጭ የአካል ማጎልመሻ ትምህርት ለማዘጋጀት የሚረዱ መሳሪያዎችም ሆኑ የሰው ሀይል የለንም አለ።  
  **HYP:** ከዚህም በላይ ምስጢሩን ወይም ሰውን የሚያቀርቡትን መሰረታዊ የትምህርት ፕሮግራም ለማሟላት የሚያስችል አማራጭ እንደሆነ ይናገራሉ። [በገጽ 20 ላይ የሚገኝ የተቀነጨበ ሀሳብ]
- **SRC:** i felt handicapped - not by my deafness or blindness - but by my intense emotional turmoil .  
  **REF:** አብዛኛውን ጊዜ ወደ ስብሰባ መሄድ ወይም አገልግሎት መውጣት የምችል ሆኖ አይሰማኝም ነበር።  
  **HYP:** በምእራባዊነት ወይም በስድስትና በስንዴት ብቻ ሳይሆን በጎዳናዬም ሆነ በጠላቶቼ ላይ ተጽእኖ አሳድሮ ነበር። [በገጽ 20 ላይ የሚገኝ የተቀነጨበ ሀሳብ]
- **SRC:** my wife and i look at each other .  
  **REF:** እኔና ባለቤቴ ተያየን።  
  **HYP:** እኔና ባለቤቴ እርስ በርስ ተመለከትኩ።

### word order

- **SRC:** the answers to those questions will be discussed in the next article .  
  **REF:** የሚቀጥለው ርእስ የእነዚህን ጥያቄዎች መልስ ይዟል።  
  **HYP:** የእነዚህ ጥያቄዎች መልስ በሚቀጥለው ርእስ ላይ ይብራራል።
- **SRC:** only joshua and caleb urged the people not to rebel out of fear , for jehovah would surely be with them .  
  **REF:** ይሆዋ ከእነርሱ ጋር እንደሚሆን በመተማመን ህዝቡ ከፍርሀት የተነሳ ማመጽ እንደሌለባቸው የተናገሩት ኢያሱና ካሌብ ብቻ ነበሩ።  
  **HYP:** ኢያሱና ካሌብ፣ ይሆዋ ከእነሱ ጋር እንደሚሆን ምንም ጥርጥር የለውም።
- **SRC:** " who really is the faithful and discreet slave whom his master appointed over his domestics ? " matt . 24 : 45 .  
  **REF:** " ጌታው በአገልጋዮቹ ላይ የሾመው ታማኝና ልባም ባሪያ በእርግጥ ማን ነው? " ማቴ 24፥ 45  
  **HYP:** (ማቴ 24: 45) ' ጌታውን የሚሾመው ታማኝና ልባም ባሪያ ማን ነው? ' ማቴ.

### named entity

- **SRC:** and there remained among the children of israel seven tribes , which had not yet received their inheritance .  
  **REF:** ከእስራኤልም ልጆች ርስት ያልተካፈሉ ሰባት ነገድ ቀርተው ነበር።  
  **HYP:** ከሰባትም ልጆች መካከል ርስታቸውን ሰባት ነገዶች አላገኙም።
- **SRC:** once your motive is clear and strong , you are ready to praise jehovah enthusiastically .  
  **REF:** ይሆዋን የምታመሰግንበት ግልጽና ጠንካራ ምክንያት ካለህ እርሱን በቅንአት ለማመስገን ዝግጁ ነህ ማለት ነው።  
  **HYP:** አላማህ ግልጽና ጠንካራ ነው።
- **SRC:** twenty - three thousand lost jehovah ' s favor shortly before joshua was to lead god ' s people into the promised land .  
  **REF:** ኢያሱ የአምላክን ህዝብ ወደ ተስፋይቱ ምድር ከማስገባቱ ጥቂት ቀደም ብሎ ሀያ ሶስት ሺህ ሰዎች የይሆዋን ሞገስ አጥተዋል።  
  **HYP:** ሀያ ሶስት ሺህ አመታት ኢያሱ የአምላክን ህዝቦች ወደ ተስፋይቱ ምድር ከመስጠት በፊት የአምላክን ሞገስ ያጡ ነበር።

### rare

- **SRC:** besides , they claimed that they had neither the facilities nor the manpower to provide an alternative physical education program .  
  **REF:** ከዚህም በላይ አማራጭ የአካል ማጎልመሻ ትምህርት ለማዘጋጀት የሚረዱ መሳሪያዎችም ሆኑ የሰው ሀይል የለንም አለ።  
  **HYP:** ከዚህም በላይ ምስጢሩን ወይም ሰውን የሚያቀርቡትን መሰረታዊ የትምህርት ፕሮግራም ለማሟላት የሚያስችል አማራጭ እንደሆነ ይናገራሉ። [በገጽ 20 ላይ የሚገኝ የተቀነጨበ ሀሳብ]
- **SRC:** as told by george warienchuck  
  **REF:** ጆርጅ ዎረንቸክ እንደተናገረው  
  **HYP:** በስፔይን ጦርነቱን እንደተናገረው
- **SRC:** or a young brother or sister who has athletic ability may find that recruiters try to entice him or her into a sports career .  
  **REF:** ወይም ደግሞ በአትሌቲክስ ዘርፍ ወጣቶችን የሚመለምሉ ሰዎች በዚህ ረገድ ጥሩ ችሎታ ያለውን አንድ ክርስቲያን ወጣት ወደ ስፖርቱ አለም እንዲገባ ሊያግባቡት ይሞክሩ ይሆናል።  
  **HYP:** ወይም ደግሞ አንድ ወጣት ወንድም ወይም እህትን ለማግኘት ሞክሩ።
