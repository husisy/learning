// index.js
import {useState} from 'react';

function Header({ title}) {
    return <h1>{title ? title : 'Default title'}</h1>;
}

export default function HomePage () {
    const names = ['name a', 'name b', 'name c'];
    const [likes, setLikes] = useState(0);
    function handleClick() {
        setLikes(likes+1);
    }
    return (
        <div>
            <Header title="Hello world" />
            <ul>
                {names.map((x) => ( <li key={x}>{x}</li> ))}
            </ul>
            <button onClick={handleClick}>Likes ({likes})</button>
        </div>
    );
}
