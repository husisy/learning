# TAI: International Atomic Time
import numpy as np

np.datetime64('2005-02-25')
np.datetime64(1, 'Y')
np.datetime64('2005-02')
np.datetime64('2005-02', 'D')
np.datetime64('2005-02-25T03:30')
np.datetime64('nat')

np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
np.array(['2001-01-01T12:00', '2002-02-03T13:56:03.172'], dtype='datetime64')

np.arange('2005-02', '2005-03', dtype='datetime64[D]')

np.datetime64('2005') == np.datetime64('2005-01-01') #True

np.timedelta64(1, 'D')
np.timedelta64(4, 'h')
np.timedelta64('nat')

np.datetime64('2009-01-01') - np.datetime64('2008-01-01') #timedelta(366,'D')
np.datetime64('2009') + np.timedelta64(20, 'D') #datetime('2009-01-21')
np.datetime64('2011-06-15T00:00') + np.timedelta64(12, 'h') #datetime('2011-06-15T12:00')
np.timedelta64(1,'W') / np.timedelta64(1,'D') #7.0 #np.float64
np.datetime64('nat') - np.datetime64('2009-01-01') #datetime('nat')
np.datetime64('2009-01-01') + np.timedelta64('nat') #datetime('nat')

np.timedelta64(np.timedelta64(1, 'D'), 'h')
# np.timedelta64(np.timedelta64(1, 'M'), 'D') #error
