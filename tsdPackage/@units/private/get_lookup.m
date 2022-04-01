function l = get_lookup()
  
  ltime = dictArray({ { 'ts', 1},
		  { 'us', 0.01}, 
		  { 'msec', 10}, 
		  { 'ms', 10},
		  { 's', 10000},
		  { 'sec', 10000},
		  { 'min', 600000},
		  { 'h', 3.6e7},
		  { 'd', 8.64e8} });

  ptime = dictArray({ { 'none', 1 } });
  
  
  
  
  l = dictArray({ { 'time', ltime},
		  { 'none', ptime} });
  
  