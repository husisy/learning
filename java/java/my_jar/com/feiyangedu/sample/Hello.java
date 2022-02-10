package com.feiyangedu.sample;

public class Hello {
    private final String name;

    public Hello(String name) {
        this.name = name;
    }

    public String hello() {
        return "Hello " + name + "!";
    }
}